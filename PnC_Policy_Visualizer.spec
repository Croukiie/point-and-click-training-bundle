# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files


sb3_data = collect_data_files('stable_baselines3')


a = Analysis(
    ['visualize_policy_web_launcher.py'],
    pathex=[],
    binaries=[],
    datas=[('src\\config', 'src\\config'), ('src\\materials', 'src\\materials'), ('src\\rl\\web', 'src\\rl\\web'), ('data\\pnc_model\\pnc_mouse_tracked_26182EVZW\\env_info.yaml', 'data\\pnc_model\\pnc_mouse_tracked_26182EVZW'), ('data\\pnc_model\\pnc_mouse_tracked_26182EVZW\\best', 'data\\pnc_model\\pnc_mouse_tracked_26182EVZW\\best')] + sb3_data,
    hiddenimports=['src.rl.sac_policy'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='PnC_Policy_Visualizer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
