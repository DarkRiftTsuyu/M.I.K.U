import shutil
import os
import sys
import subprocess
import logging

log = logging.getLogger("miku")

def _find_powershell_exe() -> str | None:
    for name in ("powershell.exe", "powershell", "pwsh.exe", "pwsh"):
        found = shutil.which(name)
        if found:
            return found
    sysroot = os.environ.get("SystemRoot", r"C:\Windows")
    legacy = os.path.join(sysroot, "System32", "WindowsPowerShell", "v1.0", "powershell.exe")
    return legacy if os.path.exists(legacy) else None



_APP_REGISTRY: list[tuple[list[str], str, str]] = [
    (["cider", "cider.exe"],          "Cider",              "Cider (Apple Music client)"),
    (["spotify", "spotify.exe"],      "Spotify",            "Spotify"),
    (["discord", "discord.exe"],      "Discord",            "Discord"),
    (["code", "code.exe", "vscode"],  "Visual Studio Code", "VS Code"),
    (["firefox", "firefox.exe"],      "Firefox",            "Firefox"),
    (["chrome", "google-chrome",
      "google-chrome-stable"],        "Google Chrome",      "Chrome"),
    (["steam", "steam.exe"],          "Steam",              "Steam"),
    (["obs", "obs-studio"],           "OBS",                "OBS Studio"),
    (["vlc"],                         "VLC",                "VLC"),
    (["slack", "slack.exe"],          "Slack",              "Slack"),
    (["notion", "notion.exe"],        "Notion",             "Notion"),
    (["obsidian", "obsidian.exe"],    "Obsidian",           "Obsidian"),
    (["terminal", "gnome-terminal",
      "konsole", "xterm"],            "Terminal",           "Terminal"),
    (["notepad", "gedit", "kate"],    "",                   "Text editor"),
    (["explorer", "nautilus",
      "thunar", "dolphin"],           "",                   "File manager"),
]


def _launch_app(app_name: str) -> str:
    target = app_name.strip().lower()

    matched_exes: list[str] = []
    matched_bundle: str = ""
    matched_label: str = target

    for exes, bundle, label in _APP_REGISTRY:
        if any(target in exe or exe in target for exe in exes):
            matched_exes = exes
            matched_bundle = bundle
            matched_label = label
            break

    if not matched_exes:
        matched_exes = [target]
        matched_label = target

    if sys.platform == "darwin" and matched_bundle:
        result = subprocess.run(
            ["open", "-a", matched_bundle],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return f"Opened {matched_label}!"

    for exe in matched_exes:
        found = shutil.which(exe)
        if found:
            try:
                if sys.platform == "win32":
                    subprocess.Popen(
                        [found],
                        creationflags=subprocess.DETACHED_PROCESS
                        | subprocess.CREATE_NEW_PROCESS_GROUP,
                    )
                else:
                    subprocess.Popen([found],
                                     stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL,
                                     start_new_session=True)
                return f"Opened {matched_label}!"
            except Exception as exc:
                log.warning("[Launcher] %s failed: %s", found, exc)

    if sys.platform == "win32":
        try:
            subprocess.Popen(
                ["explorer", f"shell:AppsFolder\\{target}"],
                creationflags=subprocess.DETACHED_PROCESS,
            )
            return f"Tried to open {matched_label} via Windows shell."
        except Exception:
            pass

    return (f"I couldn't find {matched_label} on your system. "
            f"Make sure it's installed and on your PATH.")