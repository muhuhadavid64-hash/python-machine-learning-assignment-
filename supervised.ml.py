from sklearn.linear_model import LinearRegression

hours = [[1],[2],[3],[4],[5],
        [6],[7],[8],[9],[10]]

scores= [30,40,45,55,65,70,80,85,90,95]

x = hours
y = scores

model= LinearRegression()
model.fit(x,y) 

study_hours = float(input("Enter the number of hours you studied: "))
predicted_score = model.predict([[study_hours]])
print(f"Predicted score for {study_hours} hours of study: {predicted_score[0]:.2f}")
