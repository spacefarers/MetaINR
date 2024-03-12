import torch
from CoordNet import CoordNet, SIREN,Column
from torchview import draw_graph

input_size=[16384, 3]
x = torch.randn(input_size)

model = Column()
# model = SIREN(in_features=3, out_features=1, init_features=64, num_res=3)

model_graph = draw_graph(model, input_data=x, expand_nested=True, save_graph=True, filename="new_model")
