document.addEventListener('DOMContentLoaded', () => {
  const promptInput = document.getElementById('prompt');
  const sendBtn = document.getElementById('sendBtn');
  const outputDiv = document.getElementById('output');

  sendBtn.addEventListener('click', async () => {
    const promptText = promptInput.value.trim();
    if (!promptText) return;

    sendBtn.disabled = true;
    sendBtn.textContent = 'Generating...';
    outputDiv.style.display = 'block';
    outputDiv.textContent = 'Contacting MyGPT API...';

    try {
      const response = await fetch('http://127.0.0.1:8000/api/v1/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: promptText, max_new_tokens: 50 }),
      });

      if (response.ok) {
        const data = await response.json();
        outputDiv.textContent = data.generated_text;
      } else {
        outputDiv.textContent = `Response: ${promptText} (Offline Mode)`;
      }
    } catch {
      outputDiv.textContent = `MyGPT Extension (Offline Mode):\nProcessed prompt: "${promptText}"`;
    } finally {
      sendBtn.disabled = false;
      sendBtn.textContent = 'Generate Response';
    }
  });
});
