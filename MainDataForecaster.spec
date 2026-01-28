# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all, collect_submodules
import sys
import os

# Incluir assets (logos), .env y otros archivos necesarios
datas = [
    ('ui/assets', 'ui/assets'),  # Logo.ico, IDC.ico, riopaila_logo.png
    ('.env', '.'),  # .env en la raíz del bundle
]

binaries = []

# Hidden imports esenciales
hiddenimports = [
    'sklearn.utils._weight_vector',
    'sklearn.neighbors._partition_nodes',
    'pandas',
    'numpy',
    'scipy',
    'matplotlib',
    'plotly',
    'PIL',
    'PIL.Image',
    'PIL.ImageTk',
    'tqdm',
    'openpyxl',
    'pyarrow',
]

# Collect all torch dependencies - intentar con collect_all
try:
    tmp_ret = collect_all('torch')
    datas += tmp_ret[0]
    binaries += tmp_ret[1]
    hiddenimports += tmp_ret[2]
    print("✓ Torch collected successfully")
except Exception as e:
    print(f"⚠️ Warning collecting torch: {e}")
    # Fallback: collect submodules manually
    hiddenimports += collect_submodules('torch')

# Collect all sklearn dependencies
tmp_ret = collect_all('sklearn')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

a = Analysis(
    ['launch_app.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
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
    name='MainDataForecaster',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Mostrar consola para debugging
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='ui\\assets\\IDC.ico',  # Icono del ejecutable
)
