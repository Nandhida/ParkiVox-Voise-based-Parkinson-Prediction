async function sendMessage() {
  const inputField = document.getElementById("userInput");
  const message = inputField.value.trim();
  if (!message) return;

  const chatbox = document.getElementById("chatbox");

  // Add user message
  const userMsg = document.createElement("div");
  userMsg.className = "message user";
  userMsg.textContent = "You: " + message;
  chatbox.appendChild(userMsg);

  inputField.value = "";

  // Scroll to bottom
  chatbox.scrollTop = chatbox.scrollHeight;

  // Add typing indicator
  const typingMsg = document.createElement("div");
  typingMsg.className = "message bot typing";
  typingMsg.textContent = "Bot is typing...";
  chatbox.appendChild(typingMsg);

  chatbox.scrollTop = chatbox.scrollHeight;

  try {
    // <-- CHANGE THIS URL to /gemini_chat -->
    const response = await fetch("/gemini_chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });

    const data = await response.json();

    // Remove typing indicator
    typingMsg.remove();

    // Add bot response
    const botMsg = document.createElement("div");
    botMsg.className = "message bot";
    botMsg.textContent = "Bot: " + (data.reply || "I didn't understand that. Please rephrase.");
    chatbox.appendChild(botMsg);

    chatbox.scrollTop = chatbox.scrollHeight;
  } catch (error) {
    console.error("Error:", error);

    typingMsg.remove();
    const botMsg = document.createElement("div");
    botMsg.className = "message bot";
    botMsg.textContent = "Bot: Sorry, something went wrong.";
    chatbox.appendChild(botMsg);
    chatbox.scrollTop = chatbox.scrollHeight;
  }
}
