import pickle

with open('./four_room_dataset.pkl', 'rb') as file:
    data = pickle.load(file)

print(data)