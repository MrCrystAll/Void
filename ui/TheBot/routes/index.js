var express = require('express');
var router = express.Router();
var fs = require("node:fs")
var path = require("node:path")

var rewards = [
    {"EventReward": 0.0, "FaceBallReward": -7.645799784176818, "LiuDistanceBallToGoalReward": 611.7444786635704, "VelocityBallToGoalReward": -2.7340726209858643},
    {"EventReward": 0.0, "FaceBallReward": -7.645799784176818, "LiuDistanceBallToGoalReward": 611.7444786635704, "VelocityBallToGoalReward": -2.7340726209858643},
    {"EventReward": 0.0, "FaceBallReward": -7.645799784176818, "LiuDistanceBallToGoalReward": 611.7444786635704, "VelocityBallToGoalReward": -2.7340726209858643},
    {"EventReward": 0.0, "FaceBallReward": -7.645799784176818, "LiuDistanceBallToGoalReward": 611.7444786635704, "VelocityBallToGoalReward": -2.7340726209858643},
    {"EventReward": 0.0, "FaceBallReward": -7.645799784176818, "LiuDistanceBallToGoalReward": 611.7444786635704, "VelocityBallToGoalReward": -2.7340726209858643},
    {"EventReward": 0.0, "FaceBallReward": -7.645799784176818, "LiuDistanceBallToGoalReward": 611.7444786635704, "VelocityBallToGoalReward": -2.7340726209858643},
    {"EventReward": 0.0, "FaceBallReward": -7.645799784176818, "LiuDistanceBallToGoalReward": 611.7444786635704, "VelocityBallToGoalReward": -2.7340726209858643},
    {"EventReward": 0.0, "FaceBallReward": -7.645799784176818, "LiuDistanceBallToGoalReward": 611.7444786635704, "VelocityBallToGoalReward": -2.7340726209858643},
    {"EventReward": 0.0, "FaceBallReward": -7.645799784176818, "LiuDistanceBallToGoalReward": 611.7444786635704, "VelocityBallToGoalReward": -2.7340726209858643},
    {"EventReward": 0.0, "FaceBallReward": -7.645799784176818, "LiuDistanceBallToGoalReward": 611.7444786635704, "VelocityBallToGoalReward": -2.7340726209858643},
    {"EventReward": 0.0, "FaceBallReward": -7.645799784176818, "LiuDistanceBallToGoalReward": 611.7444786635704, "VelocityBallToGoalReward": -2.7340726209858643},
    {"EventReward": 0.0, "FaceBallReward": -7.645799784176818, "LiuDistanceBallToGoalReward": 611.7444786635704, "VelocityBallToGoalReward": -2.7340726209858643}
]

fs.watchFile(path.join(__dirname, "../../rewards.json"), (curr, prev) => {
  rewards.push(JSON.parse(fs.readFileSync(path.join(__dirname, "../../rewards.json")).toString()))
})

/* GET home page. */
router.get('/', function(req, res, next) {
  res.render('index', { title: 'Void - Home' , rewards: rewards});

});

module.exports = router;
