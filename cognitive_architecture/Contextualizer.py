"""
Adds context to an ambiguous action.
"""


class Contextualizer:
    def __init__(self):
        self.lut = {
            'USE': {
                'SINK': 'WASH',
                'HOBS': 'COOK',
                'PLATE': 'EAT',
                'BOTTLE': 'SIP'
            }
        }

    def give_context(self, action, context):
        """
        Given an action and a context, it will return a contextualized action, if it exists.

        :param action: str
        :param context: str
        :return: contextualized action, str
        """
        action = action.upper()
        context = context.upper()
        row = self.lut.get(action, action)   # If action is not in the LUT, it will default to action itself
        try:
            ca = row.get(context, action)
            return ca
        except AttributeError:               # When action is not in the LUT, row will be a str and not a dict
            return action
