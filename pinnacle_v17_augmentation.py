import torch
import numpy as np
from torchvision import transforms
import random
from scipy.ndimage import gaussian_filter1d

class RamanAugmentation:
    def __init__(self, config):
        aug = config['augmentation']['raman']
        self.noise_std = aug['noise_std']
        self.shift_range = aug['shift_range']
        self.scale_range = aug['scale_range']
        self.baseline_drift = aug['baseline_drift']
        self.peak_broadening = aug.get('peak_broadening', True)
        self.probability = aug.get('probability', 0.7)
    
    def __call__(self, spectrum):
        if random.random() > self.probability:
            return spectrum
        spectrum = spectrum.clone()
        if random.random() > 0.5:
            spectrum = spectrum + torch.randn_like(spectrum) * self.noise_std
        if random.random() > 0.5:
            spectrum = torch.roll(spectrum, random.randint(-self.shift_range, self.shift_range), dims=0)
        if random.random() > 0.5:
            spectrum = spectrum * (1.0 + random.uniform(-self.scale_range, self.scale_range))
        if self.baseline_drift and random.random() > 0.5:
            x = torch.linspace(0, 1, len(spectrum), device=spectrum.device)
            spectrum = spectrum + random.uniform(-0.1, 0.1) * (x ** 2)
        if self.peak_broadening and random.random() > 0.3:
            s = spectrum.cpu().numpy()
            s = gaussian_filter1d(s, sigma=random.uniform(0.5, 1.5), mode='nearest')
            spectrum = torch.from_numpy(s).to(spectrum.device)
        return spectrum

class ImageAugmentation:
    def __init__(self, config):
        aug = config['augmentation']['image']
        self.probability = aug.get('probability', 0.5)
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=aug.get('rotation', 15)),
            transforms.ColorJitter(brightness=aug.get('brightness', 0.2), contrast=aug.get('contrast', 0.2)),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ])
    def __call__(self, image):
        return self.transform(image) if random.random() <= self.probability else image

class MixupAugmentation:
    def __init__(self, alpha=0.2, probability=0.5):
        self.alpha, self.probability = alpha, probability
    def __call__(self, x_raman, x_image, y):
        if random.random() > self.probability:
            return x_raman, x_image, y, y, 1.0
        lam = np.random.beta(self.alpha, self.alpha)
        idx = torch.randperm(x_raman.size(0)).to(x_raman.device)
        return (lam * x_raman + (1-lam) * x_raman[idx], 
                lam * x_image + (1-lam) * x_image[idx], y, y[idx], lam)

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class TestTimeAugmentation:
    def __init__(self, n_iterations=5):
        self.n_iterations, self.raman_aug, self.image_aug = n_iterations, None, None
    def set_augmentations(self, config):
        self.raman_aug, self.image_aug = RamanAugmentation(config), ImageAugmentation(config)
    def __call__(self, model, x_raman, x_image):
        model.eval()
        preds = []
        with torch.no_grad():
            logits, _, _, _ = model(x_raman, x_image)
            preds.append(torch.softmax(logits, dim=1))
            for _ in range(self.n_iterations - 1):
                x_r = torch.stack([self.raman_aug(x) for x in x_raman])
                x_i = torch.stack([self.image_aug(x) for x in x_image])
                logits, _, _, _ = model(x_r, x_i)
                preds.append(torch.softmax(logits, dim=1))
        return torch.stack(preds).mean(dim=0)

class EarlyStopping:
    def __init__(self, patience=12, min_delta=0.001):
        self.patience, self.min_delta, self.counter, self.best_score, self.early_stop = patience, min_delta, 0, None, False
    def __call__(self, val_acc):
        if self.best_score is None:
            self.best_score = val_acc
        elif val_acc < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score, self.counter = val_acc, 0
        return self.early_stop
