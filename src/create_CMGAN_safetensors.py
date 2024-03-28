from models import generator
import torch
import sys
import safetensors

model_path = sys.argv[1]

if(__name__)=="__main__":
	n_fft = 400
	print ("loading: ",model_path)
	model = generator.TSCNet(num_channel=64, num_features=n_fft // 2 + 1)
	model.load_state_dict((torch.load(model_path,map_location=torch.device('cpu'))))

	#safetensors.torch.save_file(model, 'out.st')
	weights = model.state_dict()
	safetensors.torch.save_file(weights, 'out.st')
