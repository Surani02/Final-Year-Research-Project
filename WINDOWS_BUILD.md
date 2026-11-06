# Offline Windows build

The demo is an inference-only Streamlit application. It loads the existing
PyTorch checkpoints on CPU and does not run any training code. Build on
Windows 11; PyInstaller does not produce Windows executables from Linux.

## 1. Prepare the Windows build machine

Copy this repository to Windows, preserving the `checkpoints/`, `results/`,
`web_demo/`, `agents/`, `envs/`, and `modules/` directories. Do not copy the
Linux `.venv`. Install 64-bit Python 3.11 or 3.12 once on the build machine
(the target machine will not need Python), then run PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pyinstaller
```

Use the same Python/PyTorch versions used to validate the application. If the
build machine is offline, download the matching Windows wheels on another
machine and install them with `pip install --no-index --find-links <wheel-dir>`.
Use CPU-only PyTorch wheels; CUDA is not required.

## 2. Build and test `onedir` first

From the repository root, run:

```bat
build_windows.bat
```

This creates `dist\RL_Demo\RL_Demo.exe`. Copy the complete
`dist\RL_Demo` directory to a clean Windows 11 machine (not just the EXE),
disconnect the machine from the network, and double-click `RL_Demo.exe`.
The Streamlit interface should open in the default browser at
`http://localhost:8501`. Test each environment, mode, seed, and comparison
tab that you intend to ship. Confirm that no training process or network
request is required.

The spec includes every `.pt` checkpoint in the repository and the local
Streamlit configuration. It does not include ZIP archives, generated figures,
CSV analysis outputs, or the Linux virtual environment because the inference
UI does not read them.

## 3. Optional `onefile` build

After the onedir build passes on a clean offline machine:

```bat
call .venv\Scripts\activate.bat
set RL_DEMO_ONEFILE=1
python -m PyInstaller --clean --noconfirm RL_Demo.spec
```

The result is `dist\RL_Demo.exe`. Copy only that file to a clean Windows 11
machine and double-click it. Onefile extraction can take longer and uses
temporary disk space, so ship the onedir directory if startup time or antivirus
scanning is a concern.

## Troubleshooting

- **Missing checkpoint:** verify the corresponding `.pt` file was present when
  the spec was built and rebuild after adding checkpoints.
- **Browser does not open:** browse manually to `http://localhost:8501` and
  inspect the console window for the first error.
- **Port already in use:** close the other Streamlit process or change the
  port in `web_demo/run.py` and rebuild.
- **LunarLander fails to render:** ensure the Windows `gymnasium[box2d]`
  dependencies and `pygame` installed successfully in the build environment.
