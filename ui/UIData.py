from json import JSONEncoder
from typing import Optional, Dict, NamedTuple, Any, Tuple, List


class UIData:
    class RewardFormat:
        def __init__(self, name, value, percentage):
            self.name = name
            self.value = value
            self.percentage = percentage

    class RewardFormatEncoder(JSONEncoder):
        def encode(self, o: Any) -> str:
            if isinstance(o, UIData.RewardFormat):
                return "{" \
                       f"\"name\": \"{o.name}\"," \
                       f"\"value\": {o.value}," \
                       f"\"percentage\": {o.percentage}" \
                       "}"

            if isinstance(o, (List, Tuple)):
                message = "["
                for index, elt in enumerate(o):
                    message += self.encode(elt)
                    if index != len(o) - 1:
                        message += ","
                message += "]"
                return message


    def __init__(self, state_name: Optional[str] = None, rewards: Optional[Dict] = None):
        self.state_name = state_name
        self.rewards = rewards
        self.rewards_array = []

    def __calculate(self):

        reward = self.rewards.copy()

        for key in self.rewards:
            self.rewards_array.append(
                self.RewardFormat(
                    name=key,
                    value=self.rewards[key],
                    percentage=abs(self.rewards[key]) / sum(abs(number) for number in list(reward.values()))
                )
            )

    def jsonify(self) -> str:
        if not self.state_name or self.rewards_array is None or not self.rewards:
            raise Exception("State name or rewards are None")

        self.__calculate()
        rewards = self.RewardFormatEncoder().encode(self.rewards_array)

        return "{" \
               f"\"state_name\": \"{self.state_name}\"," \
               f"\"rewards\": {rewards}" \
               "}"
