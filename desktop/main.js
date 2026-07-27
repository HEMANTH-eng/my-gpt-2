const { app, BrowserWindow, globalShortcut, ipcMain, Notification, Tray, Menu } = require('electron');
const path = require('path');

let mainWindow = null;
let tray = null;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    title: 'MyGPT Desktop Studio',
    backgroundColor: '#09090b',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: false,
      contextIsolation: true,
    },
  });

  // Load Next.js web application interface
  mainWindow.loadURL('http://localhost:3000');

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

function createTray() {
  tray = new Tray(path.join(__dirname, 'tray_icon.png'));
  const contextMenu = Menu.buildFromTemplate([
    { label: 'Show MyGPT Studio', click: () => mainWindow && mainWindow.show() },
    { label: 'Hide', click: () => mainWindow && mainWindow.hide() },
    { type: 'separator' },
    { label: 'Quit MyGPT', click: () => app.quit() },
  ]);
  tray.setToolTip('MyGPT Desktop Studio');
  tray.setContextMenu(contextMenu);
}

app.whenReady().then(() => {
  createWindow();
  try {
    createTray();
  } catch {
    // Graceful fallback if tray icon image is missing
  }

  // Register global shortcut (Ctrl+Shift+G / Cmd+Shift+G) to toggle app window
  const shortcut = process.platform === 'darwin' ? 'Command+Shift+G' : 'Control+Shift+G';
  globalShortcut.register(shortcut, () => {
    if (mainWindow) {
      if (mainWindow.isVisible()) {
        mainWindow.hide();
      } else {
        mainWindow.show();
        mainWindow.focus();
      }
    }
  });

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('will-quit', () => {
  globalShortcut.unregisterAll();
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

ipcMain.on('notify', (event, arg) => {
  if (Notification.isSupported()) {
    new Notification({ title: arg.title || 'MyGPT Desktop', body: arg.body || '' }).show();
  }
});

ipcMain.handle('get-version', () => app.getVersion());
