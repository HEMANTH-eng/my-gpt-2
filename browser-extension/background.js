chrome.runtime.onInstalled.addListener(() => {
  console.log("MyGPT Assistant Extension v2.0 Installed Successfully.");
});

// Enable side panel on extension icon click if supported
if (chrome.sidePanel && chrome.sidePanel.setPanelBehavior) {
  chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true });
}
