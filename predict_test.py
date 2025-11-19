import requests

url = "http://localhost:9696/predict"

patient_id = 'xyz-123'
patient = {
    "pregnancies": 2,
    "glucose": 120,
    "blood_pressure": 70,
    "skin_thickness": 20,
    "insulin": 79,
    "bmi": 28.0,
    "diabetes_pedigree_function": 0.5,
    "age": 32
}

response = requests.post(url, json=patient).json()

print(response)

if response['diabetes'] == True:
    print('send alert to patient %s' % patient_id)
else:
    print('no alert for patient %s' % patient_id)
