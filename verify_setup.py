import yaml, os
def verify():
    print("="*70)
    with open('config.yaml') as f:
        c = yaml.safe_load(f)
    checks = {'epochs≥50': c.get('epochs')>=50, 'warmup≥10': c.get('arch_warmup')>=10,
              'lr_arch': 'lr_arch' in c, 'aug': c.get('augmentation',{}).get('enabled'),
              'mixup': c.get('mixup',{}).get('enabled'), 'TTA': c.get('test_time_augmentation',{}).get('enabled')}
    for k,v in checks.items(): print(f"   {'✅' if v else '❌'} {k}")
    try:
        from pinnacle_v17_augmentation import RamanAugmentation
        print("   ✅ Imports OK")
    except: print("   ❌ Import failed")
    print("="*70)
if __name__=='__main__': verify()
