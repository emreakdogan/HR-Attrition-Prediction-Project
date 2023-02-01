from fastapi import FastAPI
from ml import predict
from preprocess import preprocess
from sample import Sample


app = FastAPI()


@app.post("/predict/")
def read_items(sample: Sample) -> int:
    sample_dict = sample.__dict__
    preprocessed_sample = preprocess(sample_dict)
    prediction = predict(preprocessed_sample)

    return prediction


@app.get("/whoami")
def whoami() -> str:
    # TODO
    isim = "EMRE"
    soyisim = "AKDOĞAN"
    mail = "emreakdgn00@gmail.com"
    
    person_card = {
        "isim": isim,
        "soyisim": soyisim,
        "mail": mail
    }

    return person_card


@app.get("/model_card")
def model_card() -> str:
    # TODO

    model_card = {
        'model_name': 'EA',
        'model_description': '',
        'model_version': 'V01.04.22',
        'model_author': 'EMRE AKDOĞAN',
        'model_author_mail': 'emreakdgn00@gmail.com',
        'model_creation_date': '',
        'model_last_update_date': '01.04.2022',
        'required_parameters_list': '',
        'required_parameters_descriptions': '',
    }

    return model_card


import uvicorn

if __name__ == "__main__":
    uvicorn.run("main:app",host="127.0.0.1",port=5000,log_level="info")








