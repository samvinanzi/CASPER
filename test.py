from FocusBelief import FocusBelief
from DataframeFactory import DataFrameFactory


factory = DataFrameFactory()
factory.reload_data()
focus = FocusBelief('human')

for i in range(35):
    new_episode = factory.build_episode(i)
    print(new_episode)
    if new_episode is None:
        continue
    else:
        objects = new_episode.get_objects_for('human')
        for object in objects:
            focus.add(object)
        if focus.has_confident_prediction():
            prediction = focus.get_top_n_items(1)
        else:
            prediction = None
        print("Prediction: *{0}*".format(prediction))
        focus.print_probabilities()
