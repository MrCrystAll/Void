class Reward {
    constructor(name, value, percentage) {
        this.name = name
        this.value = value
        this.percentage = percentage
    }

    getFormattedName(){
        //return this.name.substring(0, this.name.length - 6).replace(/([A-Z])/g, ' $1').trim()
        return this.name.replace("Reward","").replace(/([A-Z])/g, ' $1').trim()
    }


}

class Episode{
    constructor(stateName, rewards) {
        this.stateName = stateName
        this.rewards = rewards
    }
}

function getAverageValueOf(arr){
    console.log(arr)
    return (arr.reduce((r, a) => r + a.rewards[rew.name].value, 0) / data.length).toFixed(2)
}

module.exports = {
    "Reward": Reward,
    "Episode": Episode
}