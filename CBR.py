import pandas as pd
import json

class CaseBaseReasoning():
    def __init__(self, weight_range = [1,1,1,2,2,1,1,1,1,1,1,1], thresholds = 0.9):
        self.occupation_mapping = {
            "Accountant": {
                "Accountant": 1.0,
                "Manager": 0.6,
                "Engineer": 0.4,
                "Scientist": 0.3,
                "Software Engineer": 0.3,
                "Teacher": 0.4,
                "Lawyer": 0.5,
                "Salesperson": 0.3,
                "Sales Representative": 0.3,
                "Doctor": 0.2,
                "Nurse": 0.2,
            },
            "Doctor": {
                "Doctor": 1.0,
                "Nurse": 0.8,
                "Scientist": 0.6,
                "Teacher": 0.5,
                "Lawyer": 0.2,
                "Engineer": 0.2,
                "Software Engineer": 0.2,
                "Accountant": 0.2,
                "Salesperson": 0.1,
                "Sales Representative": 0.1,
                "Manager": 0.3,
            },
            "Engineer": {
                "Engineer": 1.0,
                "Software Engineer": 0.9,
                "Scientist": 0.7,
                "Manager": 0.6,
                "Accountant": 0.4,
                "Teacher": 0.4,
                "Doctor": 0.2,
                "Nurse": 0.2,
                "Salesperson": 0.3,
                "Sales Representative": 0.3,
                "Lawyer": 0.2,
            },
            "Lawyer": {
                "Lawyer": 1.0,
                "Manager": 0.6,
                "Accountant": 0.5,
                "Teacher": 0.4,
                "Salesperson": 0.3,
                "Sales Representative": 0.3,
                "Engineer": 0.2,
                "Doctor": 0.2,
                "Nurse": 0.2,
                "Scientist": 0.3,
                "Software Engineer": 0.3,
            },
            "Manager": {
                "Manager": 1.0,
                "Accountant": 0.6,
                "Engineer": 0.6,
                "Salesperson": 0.7,
                "Sales Representative": 0.7,
                "Lawyer": 0.6,
                "Teacher": 0.5,
                "Software Engineer": 0.5,
                "Scientist": 0.5,
                "Doctor": 0.3,
                "Nurse": 0.3,
            },
            "Nurse": {
                "Nurse": 1.0,
                "Doctor": 0.8,
                "Scientist": 0.5,
                "Teacher": 0.5,
                "Manager": 0.3,
                "Engineer": 0.2,
                "Software Engineer": 0.2,
                "Accountant": 0.2,
                "Salesperson": 0.1,
                "Sales Representative": 0.1,
                "Lawyer": 0.2,
            },
            "Sales Representative": {
                "Sales Representative": 1.0,
                "Salesperson": 0.95,
                "Manager": 0.7,
                "Teacher": 0.4,
                "Engineer": 0.3,
                "Software Engineer": 0.3,
                "Lawyer": 0.3,
                "Accountant": 0.3,
                "Doctor": 0.1,
                "Nurse": 0.1,
                "Scientist": 0.2,
            },
            "Salesperson": {
                "Salesperson": 1.0,
                "Sales Representative": 0.95,
                "Manager": 0.7,
                "Teacher": 0.4,
                "Engineer": 0.3,
                "Software Engineer": 0.3,
                "Lawyer": 0.3,
                "Accountant": 0.3,
                "Doctor": 0.1,
                "Nurse": 0.1,
                "Scientist": 0.2,
            },
            "Scientist": {
                "Scientist": 1.0,
                "Engineer": 0.7,
                "Software Engineer": 0.7,
                "Doctor": 0.6,
                "Nurse": 0.5,
                "Teacher": 0.4,
                "Manager": 0.5,
                "Accountant": 0.3,
                "Lawyer": 0.3,
                "Salesperson": 0.2,
                "Sales Representative": 0.2,
            },
            "Software Engineer": {
                "Software Engineer": 1.0,
                "Engineer": 0.9,
                "Scientist": 0.7,
                "Manager": 0.5,
                "Teacher": 0.4,
                "Accountant": 0.3,
                "Lawyer": 0.3,
                "Salesperson": 0.3,
                "Sales Representative": 0.3,
                "Doctor": 0.2,
                "Nurse": 0.2,
            },
            "Teacher": {
                "Teacher": 1.0,
                "Scientist": 0.4,
                "Doctor": 0.5,
                "Nurse": 0.5,
                "Lawyer": 0.4,
                "Engineer": 0.4,
                "Software Engineer": 0.4,
                "Accountant": 0.4,
                "Manager": 0.5,
                "Salesperson": 0.4,
                "Sales Representative": 0.4,
            },
        }

        columns = ['Gender', 'Age', 'Occupation', 'Sleep Duration', 'Quality of Sleep',
               'Physical Activity Level', 'Stress Level', 'BMI Category',
               'Sistole', 'Diastole', 'Daily Steps', 'Sleep Disorder']
        self.database = pd.DataFrame(columns=columns)
        self.index = 0
        self.JSON_PATH = "CaseBase.json"
        self.weight_range = weight_range
        self.thresholds = float(thresholds)
        self.trainData()

    def trainData(self):
        df = pd.read_csv("Sleep_health_and_lifestyle_dataset_Preproc.csv")
        df["Sleep Disorder"] = df['Sleep Disorder'].fillna('Normal')

        target_col = 'Sleep Disorder'
        feature_cols = df.columns.drop(target_col)
        self.features = feature_cols.tolist()

        self.df_encoded = df.copy()
        self.df_encoded['Gender'] = self.df_encoded['Gender'].map({'Male': 0, 'Female': 1})
        self.df_encoded['BMI Category'] = self.df_encoded['BMI Category'].map({'Normal': 0, 'Overweight': 1, 'Obese': 2})
        self.df_encoded['Sistole'] = df['Sistole'].astype(int)
        self.df_encoded['Diastole'] = df['Diastole'].astype(int)

        label_map = {'Normal': 0, 'Sleep Apnea': 1, 'Insomnia': 2, None: 0}
        self.df_encoded['Sleep Disorder'] = self.df_encoded['Sleep Disorder'].replace('None', 'Normal')
        self.df_encoded[target_col] = self.df_encoded[target_col].map(label_map)

        for row in self.df_encoded.iterrows():
            # print(row[1])
            self.newCase(row[1], "Train")
        self.saveToJson()

    def newCase(self, row, State = "Train"):
        if State == "Final":
            row['Gender'] = row['Gender'].map({'Male': 0, 'Female': 1})
            row['BMI Category'] = row['BMI Category'].map({'Normal': 0, 'Overweight': 1, 'Obese': 2})
            row['Sistole'] = row['Sistole'].astype(int)
            row['Diastole'] = row['Diastole'].astype(int)
        # print(self.database.head())
        if self.index == 0 and State == "Train":
            self.retain(row)
        else:
            if State == "Final":
                input_row = row.iloc[0]
                data = self.reuse(input_row, State)
                if data == 0:
                    return "Normal"
                elif data == 1:
                    return "Sleep Apnea"
                elif data == 2:
                    return "Insomnia"
                else:
                    return "Unknown"
            else:
                return self.reuse(row, State)

    def retain(self, row):
        new_row = pd.DataFrame([row])

        # print(new_row.head())
        self.database = pd.concat([self.database, new_row], ignore_index=True)
        self.index += 1


    def reuse(self, row, state):
        idx, sim = self.retrieve(row)
        sim = float(sim)
        case = self.database.iloc[idx]
        predicted_label = case["Sleep Disorder"]
        # print (sim, type(sim), self.thresholds, type(self.thresholds), self.thresholds <= sim)
        if state != "Train":
            if state == "Validation":
                return case["Sleep Disorder"]
            elif self.thresholds >= sim:
                self.retain(row)
                print(f"Most similar case found at index {idx} with similarity {sim:.4f}, Retaining Case")
            else:
                print(f"Most similar case found at index {idx} with similarity {sim:.4f}")
                print("Predicted Sleep Disorder:", case["Sleep Disorder"])
                return case["Sleep Disorder"]
        else:
            true_label = row["Sleep Disorder"]
            if self.thresholds >= sim:
                self.retain(row)
            elif predicted_label != true_label:
                self.retain(row)

    def revise(self, row):
        print("Revise step - currently doing nothing.")
        self.retain(row)

    def retrieve(self, row):
        similarities = []
        input_row = row
        db_df = self.database

        for idx, case in db_df.iterrows():
            sim = self.calculate_similarity(input_row, case, self.weight_range)
            similarities.append((idx, sim))
        # print(similarities)
        similarities.sort(key=lambda x: x[1], reverse=True)
        # print (similarities)
        return similarities [0]

    def calculate_similarity(self, row1, row2, weights):
        # print("Similarity ",type(row1),type(row2))
        sim = 0
        total_weight = sum(weights)
        for i, col in enumerate(self.features):
            w = weights[i-1]
            # print (w,col,i, row1[col],type(row1[col]), row2[col],type(row2[col]))
            if col == "Occupation":
                s = self.occupation_mapping.get(row1[col], {}).get(row2[col], 0)
            elif col == "Gender":
                s = 1.0 if row1[col] == row2[col] else 0.0
            elif col == "BMI Category":
                s = 1.0 - abs(row1[col] - row2[col]) / 2.0
            else:
                try:
                    max_val = self.df_encoded[col].max()
                    s = 1.0 - abs(row1[col] - row2[col]) / max_val
                except Exception as e:
                    print  (e)
                    print(col)
                    s = 0
            sim += w * s
            # print(s)
        # print (self.thresholds, sim, total_weight, weights)
        return sim / total_weight if total_weight != 0 else 0

    def saveToJson(self):
        df = self.database
        data_dict = df.to_dict(orient="index")
        dictFinal = dict()
        dictFinal["Weights"] = self.weight_range
        dictFinal["Thresholds"] = self.thresholds
        dictFinal["Cases"] = data_dict
        with open(self.JSON_PATH, "w") as f:
            json.dump(dictFinal, f, indent=4)

    def getWeights(self):
        return self.weight_range

    def getThresholds(self):
        return self.thresholds

    def getDF(self):
        return self.database


if __name__ == "__main__":
    cbr = CaseBaseReasoning()
    print (cbr.getWeights())
