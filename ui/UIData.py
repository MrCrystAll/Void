from json import JSONEncoder
from typing import Optional, Dict


class UIData:
    def __init__(self, state_name: Optional[str] = None, rewards: Optional[Dict] = None):
        self.state_name = state_name
        self.rewards = rewards

    def __calculate(self):

        reward = self.rewards.copy()

        for key in self.rewards:
            self.rewards[key] = {
                "value": self.rewards[key],
                "percentage": abs(self.rewards[key]) / sum(abs(number) for number in list(reward.values()))
            }

    def jsonify(self) -> str:
        if not self.state_name or not self.rewards:
            raise Exception("State name or rewards are None")

        self.__calculate()
        rewards = JSONEncoder().encode(self.rewards)

        return "{" \
               f"\"state_name\": \"{self.state_name}\"," \
               f"\"rewards\": {rewards}" \
               "}"
