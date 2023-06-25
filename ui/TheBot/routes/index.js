var express = require('express');
var router = express.Router();
var fs = require("node:fs")
var path = require("node:path")

var data = [
]

var calculated_data = [

]

const bot_name = "Void"
const page_name = "Home"

fs.watchFile(path.join(__dirname, "../../data.json"), (curr, prev) => {
  let new_data = JSON.parse(fs.readFileSync(path.join(__dirname, "../../data.json")).toString())
  data.push(new_data)
  calculated_data.push()
  console.log(data)
})

/* GET home page. */
router.get('/', function(req, res, next) {
  res.render('index', {bot_name: bot_name, title: bot_name + " - " + page_name , data: data});
  console.log(data)

});

module.exports = router;
