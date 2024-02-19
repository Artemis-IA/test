from unittest import TestCase, main
from fastapi.testclient import TestClient
from api import app
import pickle
import os

# Création du client de test
client = TestClient(app)

class TestAssertions(TestCase):
    def test_assertEqual(self):
        self.assertEqual(1, 1)

    def test_assertNotEqual(self):
        self.assertNotEqual(1, 2)
        
    def test_assertIn(self):
        self.assertIn(1, [1, 2, 3])

    def test_assertNotIn(self):
        self.assertNotIn(4, [1, 2, 3])

    def test_assertIs(self):
        a = 1
        b = a
        self.assertIs(a, b)

    def test_assertIsNot(self):
        self.assertIsNot(1, 2)

    def test_assertTrue(self):
        self.assertTrue(True)

    def test_assertFalse(self):
        self.assertFalse(False)

    def test_assertIsNone(self):
        self.assertIsNone(None)

    def test_assertIsNotNone(self):
        self.assertIsNotNone(1)
        
    def test_assertIsInstance(self):
        self.assertIsInstance(1, int)

    def test_assertNotIsInstance(self):
        self.assertNotIsInstance("1", int)

    def test_assertRaises(self):
        with self.assertRaises(ZeroDivisionError):
            result = 1/0

    def test_assertRaisesRegex(self):
        with self.assertRaisesRegex(ZeroDivisionError, "division by zero"):
            result = 1/0

class TestDevEnvironment(TestCase):
    # Vérifie que les fichiers essentiels sont présents
    def test_files_presence(self):
        required_files = ['api.py', 'requirements.txt', '.gitignore']
        list_files = os.listdir()
        for file_name in required_files:
            self.assertIn(file_name, list_files)

class TestAPI(TestCase):
    # Vérifie que l'API est bien lancée
    def test_api_root(self):
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("message", response.json())

    # Vérifie le endpoint root
    def test_root(self):
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Bienvenue sur l'API de prédiction", response.json().get("message"))

    #Vérifier le endpoint predict1
    def test_predict1(self):
        # Données d'entrée de test pour predict1
        test_data = {
            "Gender": 1,
            "Age": 30,
            "PhysicalActivityLevel": 2,
            "HearthRate": 70,
            "DailySteps": 5000,
            "BloodPressure_high": 120,
            "BloodPressure_low": 80
        }
        response = client.post("/predict1", json=test_data)
        self.assertEqual(response.status_code, 200)
        # Ici, ajustez en fonction de la sortie attendue de votre modèle
        self.assertIn("prediction", response.json())
        # Ajoutez des assertions supplémentaires ici si nécessaire

    
    def test_predict2(self):
        # Données d'entrée de test pour predict2
        test_data = {
            "PhysicalActivityLevel": 2,
            "HearthRate": 70,
            "DailySteps": 5000
        }
        response = client.post("/predict2", json=test_data)
        self.assertEqual(response.status_code, 200)
        # Ici, ajustez en fonction de la sortie attendue de votre modèle
        self.assertIn("prediction", response.json())
        # Ajoutez des assertions supplémentaires ici si nécessaire

class TestModel(TestCase):
    # Vérifie que les modèles sont présents et bien chargés
    def test_model_loading(self):
        try:
            with open('model_1.pkl', 'rb') as file:
                model1 = pickle.load(file)
            self.assertIsNotNone(model1)
        except FileNotFoundError:
            self.fail("model_1.pkl is missing")

        try:
            with open('model_2.pkl', 'rb') as file:
                model2 = pickle.load(file)
            self.assertIsNotNone(model2)
        except FileNotFoundError:
            self.fail("model_2.pkl is missing")

# Démarrage des tests
if __name__ == "__main__":
    main(verbosity=2)


# # Import des librairies
# from unittest import TestCase, main
# from fastapi.testclient import TestClient
# from api import app
# import pickle
# from unittest import TestCase, main,assertEqual, assertnonEqual, assertIn, assertNotIn, assertIs, assertIsNot, assertTrue, assertFalse, assertIsNone, assertIsNotNone, assertIsInstance, assertNotIsInstance, assertRaises, assertRaisesRegex


# # assertEqual(a, b) : Vérifie si a est égal à b.
# assertEqual(1, 1)


# # assertNotEqual(a, b) : Vérifie si a est différent de b.
# assertnonEqual(1, 2)
        
# # assertIn(a, b) : Vérifie si a est dans b.
# assertIn(1, [1, 2, 3])

# # assertNotIn(a, b) : Vérifie si a n'est pas dans b.
# assertNotIn(4, [1, 2, 3])

# # assertIs(a, b) : Vérifie si a est b.
# assertIs(1, 1)

# # assertIsNot(a, b) : Vérifie si a n'est pas b.
# assertIsNot(1, 2)

# # assertTrue(x) : Vérifie si x est vrai.
# assertTrue(1)

# # assertFalse(x) : Vérifie si x est faux.
# assertFalse(0)

# # assertIsNone(x) : Vérifie si x est None.
# assertIsNone(None)

# # assertIsNotNone(x) : Vérifie si x n'est pas None.
# assertIsNotNone(1)
        
# # assertIsInstance(a, b) : Vérifie si a est une instance de b.
# assertIsInstance(1, int)

# # assertNotIsInstance(a, b) : Vérifie si a n'est pas une instance de b.
# assertNotIsInstance(1, str)

# # assertRaises(exc, fun, *args, **kwargs) : Vérifie si fun(*args, **kwargs) lève une exception de type exc.
# assertRaises(ZeroDivisionError, lambda: 1/0)

# # assertRaisesRegex(exc, r, fun, *args, **kwargs) : Vérifie si fun(*args, **kwargs) lève une exception de type exc et dont le message correspond à l'expression régulière r.
# assertRaisesRegex(ZeroDivisionError, "division by zero", lambda: 1/0)

# # Tests unitaire de l'environnement de développement
# class TestDev(TestCase):

#     # Vérifie que les fichiers sont présents
#     def test_files(self):
#         import os
#         list_files = os.listdir()
#         assertIn('README.md', list_files)
#         assertIn('api.py', list_files)


#     # Vérifie que les requirements sont présents
#     def test_requirements(self):
#         assertIn('requirements.txt', list_files)

#     # Vérifie que le gitignore est présent
#     def test_gitignore(self):
#         assertIn('.gitignore', list_files)

# # Création du client de test
# client = TestClient(app)

# # Tests unitaire de l'API
# class TestAPI(TestCase):
#     # Vérifie que l'API est bien lancée
#     def test_api(self):
#         response = client.get("/")
#         assertEqual(response.status_code, 200)
#         assertIn("Bienvenue sur l'API de prédiction", response.json().get("message"))



#     # Vérifie le endpoint hello_you
#     def test_hello_you(self):
#         response = client.get("/hello_you?name=Jean")
#         assertEqual(response.status_code, 200)
#         assertIn("Hello Jean", response.json().get("message"))
    

#     # Vérifie le endpoint predict  
#     def test_predict(self):
#         response = client.post("/predict", json={})
#         pass

# # Test du modèle individuellement
# #class TestModel(TestCase):
# class TestModel(TestCase):
#     # Vérifie que le modèle est bien présent
#     def test_model(self):
#         with open('model_1.pkl', 'rb') as file:
#             model1 = pickle.load(file)
#         with open('model_2.pkl', 'rb') as file:
#             model2 = pickle.load(file)
#         assertIsNotNone(model1)
#         assertIsNotNone(model2)
    

#     # Vérifie que le modèle est bien chargé
#     def test_model_loaded(self):
#         with open('model_1.pkl', 'rb') as file:
#             model1 = pickle.load(file)
#         with open('model_2.pkl', 'rb') as file:
#             model2 = pickle.load(file)
#         assertIsInstance(model1, object)
#         assertIsInstance(model2, object)    

# # Démarrage des tests
# if __name__== "__main__" :
#     main(
#         verbosity=2,
#     )
