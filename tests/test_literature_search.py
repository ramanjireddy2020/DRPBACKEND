# import requests
# base_url = "http://172.25.74.92:8000/"
# upload_csv = f"{base_url}/api/v1/upload-protien-csv/"

# headers = {
#     'accept': 'application/json',
#     # requests won't add a boundary if this header is set when you pass files=
#     # 'Content-Type': 'multipart/form-data',
# }

# # files = {
# #     'file': ('sample.csv', open('sample.csv', 'rb'), 'text/csv'),
# # }

# # response = requests.post(upload_csv, headers=headers, files=files)
# # resp_dict = response.json()
# # akw_list = list(resp_dict.keys())
# articles = ['AcrAB-TolC efflux pump', 'AdeABC efflux pump', 'MexAB-OprM efflux pump', 'CmeABC efflux pump', 'SmeABC efflux pump', 'norA efflux pump', 'tet40 efflux pump', 'Staphylococcus aureus LmrS', 'emrB efflux pump', 'qacG efflux pump']
# params = {'article_keywords': articles,'search_keywords': ['inhibition','small molecule']}
# response2 = requests.get('http://172.25.74.92:8000/api/v1/search-keywords/', params=params, headers=headers)
# resp_dict = response2.json()