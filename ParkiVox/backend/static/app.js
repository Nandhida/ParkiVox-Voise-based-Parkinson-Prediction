$(document).ready(function() {
  // Navigation
  $(".nav-link").click(function(e) {
    e.preventDefault();
    $(".nav-link").removeClass("active");
    $(this).addClass("active");

    $(".tab-content").removeClass("active");
    $("#" + $(this).data("tab")).addClass("active");
  });

  // Typing test
  $("#submitTyping").click(function() {
    $.post("/typing_test", { text: $("#typingBox").val() }, function(data) {
      $("#typingResult").html("<pre>" + JSON.stringify(data, null, 2) + "</pre>");
    });
  });

  // Voice test
  $("#submitVoice").click(function() {
    let file = $("#voiceFile")[0].files[0];
    let formData = new FormData();
    formData.append("file", file);

    $.ajax({
      url: "/voice_test",
      method: "POST",
      data: formData,
      processData: false,
      contentType: false,
      success: function(data) {
        $("#voiceResult").html("<pre>" + JSON.stringify(data, null, 2) + "</pre>");
      }
    });
  });

  // Chatbot
  $("#sendChat").click(function() {
    let msg = $("#chatInput").val();
    $("#chatbox").append("<div class='user'>You: " + msg + "</div>");
    $("#chatInput").val("");

    $.post("/chatbot", { message: msg }, function(data) {
      $("#chatbox").append("<div class='bot'>Bot: " + data.reply + "</div>");
      $("#chatbox").scrollTop($("#chatbox")[0].scrollHeight);
    });
  });
});
