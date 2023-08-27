var express = require('express');
var router = express.Router();
var fs = require("node:fs")
var path = require("node:path")

var data = []

const utils = require("../public/javascripts/episode_result")
const classes = require("../public/javascripts/utils")

const bot_name = "Void"
const page_name = "Home"

const mock_data = {
    "state_name": "AerialBallState",
    "rewards": [{"name": "EventReward", "value": 0.0, "percentage": 0.0}, {
        "name": "GoalScoreSpeed",
        "value": 0.0,
        "percentage": 0.0
    }, {
        "name": "SaveBoostReward",
        "value": 10.983050718624385,
        "percentage": 0.24290832263230297
    }, {"name": "BoostPickupReward", "value": 1.5, "percentage": 0.03317497963754195}, {
        "name": "KickoffReward_MMR",
        "value": 0.0,
        "percentage": 0.0
    }, {"name": "AerialReward", "value": 0.0, "percentage": 0.0}, {
        "name": "BumpReward",
        "value": 10.5,
        "percentage": 0.23222485746279364
    }, {
        "name": "WallReward",
        "value": 0.040383301686808544,
        "percentage": 0.0008931434741043913
    }, {
        "name": "FlipReward",
        "value": 0.18206665625939,
        "percentage": 0.004026705076053742
    }, {
        "name": "ClosestDistToBallReward",
        "value": 0.7092972340374116,
        "percentage": 0.01568728086410397
    }, {"name": "WasteSpeedReward", "value": -21.300000000000168, "percentage": 0.4710847108530994}]
}

let rewards = []
mock_data.rewards.forEach(elt => {
    rewards.push(new classes.Reward(
        elt.name,
        elt.value,
        elt.percentage
    ))
})

let episode = new classes.Episode(mock_data.state_name, rewards)
let formatted_mock_data = [episode, episode]
console.log(formatted_mock_data[0].rewards[0].getFormattedName())

fs.watchFile(path.join(__dirname, "../../data.json"), (curr, prev) => {
    let new_data = JSON.parse(fs.readFileSync(path.join(__dirname, "../../data.json")).toString())


    data.push(episode)
})

/* GET home page. */
router.get('/', function (req, res, next) {
    res.render('index', {bot_name: bot_name, title: bot_name + " - " + page_name, data: formatted_mock_data, utils: utils});
    console.log(mock_data)

});

module.exports = router;
