console.log("Yeah")

function display_episode_name(episode){
    console.log(episode)
    return episode.stateName
}

module.exports = {
    "display_episode_name": display_episode_name
}