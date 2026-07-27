import json
from pathlib import Path
import pytest


def test_mobile_app_manifest_and_config():
  mobile_dir = Path("mobile")
  pkg_file = mobile_dir / "package.json"
  app_file = mobile_dir / "App.tsx"

  assert pkg_file.exists()
  assert app_file.exists()

  pkg_data = json.loads(pkg_file.read_text())
  assert pkg_data["name"] == "mygpt-mobile"
  assert "react-native" in pkg_data["dependencies"]


def test_desktop_app_manifest_and_scripts():
  desktop_dir = Path("desktop")
  pkg_file = desktop_dir / "package.json"
  main_file = desktop_dir / "main.js"
  preload_file = desktop_dir / "preload.js"

  assert pkg_file.exists()
  assert main_file.exists()
  assert preload_file.exists()

  pkg_data = json.loads(pkg_file.read_text())
  assert pkg_data["name"] == "mygpt-desktop"
  assert pkg_data["main"] == "main.js"

  main_code = main_file.read_text()
  assert "BrowserWindow" in main_code
  assert "globalShortcut" in main_code


def test_browser_extension_manifest_v3():
  ext_dir = Path("browser-extension")
  manifest_file = ext_dir / "manifest.json"
  bg_file = ext_dir / "background.js"
  popup_html = ext_dir / "popup.html"
  popup_js = ext_dir / "popup.js"

  assert manifest_file.exists()
  assert bg_file.exists()
  assert popup_html.exists()
  assert popup_js.exists()

  manifest = json.loads(manifest_file.read_text())
  assert manifest["manifest_version"] == 3
  assert manifest["name"] == "MyGPT Assistant"
  assert "sidePanel" in manifest["permissions"]
