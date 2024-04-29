import pickle

with open('datasets/four_room_dataset.pkl', 'rb') as file:
    data = pickle.load(file)

print(data)
