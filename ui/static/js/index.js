$(document).ready(function() {
    var socket = io.connect(location.protocol + '//' + document.domain + ':' + location.port + "/")

    socket.on("reward_change", (data) => {
        console.log(data)
    })
})