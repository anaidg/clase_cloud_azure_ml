# 07-model-registration-azure.py
from azureml.core import Workspace
from azureml.core import Model

if __name__ == "__main__":
    ws = Workspace.from_config(path='./.azureml',_file_name='config.json')

    model = Model.register(model_name='gestion_cartera_model',
                           tags={'area': 'udea_training'},
                           model_path='outputs/gestion_cartera.pkl',
                           workspace = ws)
    print(model.name, model.id, model.version, sep='\t')