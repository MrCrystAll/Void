$(document).ready(function() {
    var socket = io().connect("http://localhost:5000/");

    socket.on("send_model", function(msg){
        console.log(msg)
    })
})