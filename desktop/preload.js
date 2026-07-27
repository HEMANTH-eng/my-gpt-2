const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  sendNotification: (title, body) => ipcRenderer.send('notify', { title, body }),
  toggleWindow: () => ipcRenderer.send('toggle-window'),
  getAppVersion: () => ipcRenderer.invoke('get-version'),
});
