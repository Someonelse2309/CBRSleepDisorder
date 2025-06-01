import streamlit as st
from CBR import CaseBaseReasoning
import pandas as pd


def startCBR(weight_range = [1,1,1,2,2,1,1,1,1,1,1,1], thresholds = 1):
    caseBaseReasoning = CaseBaseReasoning(weight_range, thresholds)
    weights = caseBaseReasoning.getWeights()
    thresholds = caseBaseReasoning.getThresholds()
    config = {"Gender": weights[0],
                "Age": weights[0],
                "Occupation": weights[0],
                "Sleep Duration": weights[0],
                "Quality of Sleep": weights[0],
                "Physical Activity Level": weights[0],
                "Stress Level": weights[0],
                "BMI Category": weights[0],
                "Sistole": weights[0],
                "Diastole": weights[0],
                "Daily Steps": weights[0],
                "Threshold" : thresholds
             }

    return caseBaseReasoning, config

caseBaseReasoning, config = startCBR()

st.title("Case Base Reasoning For Sleep Disorder Detection")

tab1, tab2, tab3, tab4 = st.tabs(["Program", "Comparison Data", "All Data", "Setting"])

with st.sidebar:
    st.header("Anggota Kelompok:")
    st.write("""
    1. Nicholas Tanugroho - 71220845
    2. Vincen Imannuel - 71220856
    3. Nicholas Dwinata - 71220869
    """)

with tab1:
    st.header("Main Program")
    genderInp = st.selectbox("Gender",
                             tuple(sorted(("Male", "Female"))),
                             index=None,
                             placeholder="Enter your gender",
                             help="The gender of the person.",
                             )
    ageInp = st.number_input("Age",
                               min_value=27,
                               max_value=59,
                               step=1,
                               value=None,
                               placeholder="Enter your Age",
                               help="The age of the person in years.",
                               )
    occupationInp = st.selectbox("Occupation",
                                 tuple(sorted(("Accountant","Manager","Engineer","Scientist","Software Engineer","Teacher","Lawyer","Salesperson","Sales Representative","Doctor","Nurse"))),
                                 index=None,
                                 placeholder="Enter your Occupation",
                                 help="The occupation or profession of the person.",
                                 )
    sleepDurationInp = st.number_input("Sleep Duration",
                                       min_value=5.8,
                                       max_value=8.5,
                                       step=0.1,
                                       value=None,
                                       placeholder="Enter your Sleep Duration",
                                       help="The number of hours the person sleeps per day."
                                       )
    qualityOfSleepInp = st.number_input("Quality of Sleep",
                                       min_value=5.8,
                                       max_value=8.5,
                                       step=0.1,
                                       value=None,
                                       placeholder="Enter your quality of sleep score",
                                       help="A subjective rating of the quality of sleep, ranging from 1 to 10. On the development stage the minimum is 5.8 and the maximum is 8.5.",
                                       )
    physicalActivityInp = st.number_input("Physical Activity Level",
                                         min_value=30,
                                         max_value=90,
                                         step=1,
                                         value=None,
                                         placeholder="Enter your physical activity Level",
                                         help="The number of minutes the person engages in physical activity daily. On the development stage the minimum is 30 Minute(s) and the maximum is 90 Minute(s).",
                                         )
    stressLevelInp = st.number_input("Stress Level",
                                     min_value=3,
                                     max_value=8,
                                     step=1,
                                     value=None,
                                     placeholder="Enter your stress Level",
                                     help="A subjective rating of the stress level experienced by the person, ranging from 1 to 10. On the development stage the minimum is 3 and the maximum is 8.",
                                     )
    BMICategoryInp = st.selectbox("BMI Category",
                             tuple(sorted(("Normal", "Obese", "Overweight"))),
                             index=None,
                             placeholder="Enter your BMI Category",
                             help="The BMI category of the person (e.g., Underweight, Normal, Overweight). On the development stage the BMI Category won't include the underweight class",
                             )
    sistoleInp = st.number_input("Systole",
                                 min_value=115,
                                 max_value=142,
                                 step=1,
                                 value=None,
                                 placeholder="Enter your systole",
                                 help="The systole of the blood pressure measurement of the person. On the development stage the minimum is 115 and the maximum is 142.",
                                 )
    diastoleInp = st.number_input("Diastole",
                                 min_value=75,
                                 max_value=95,
                                 step=1,
                                 value=None,
                                 placeholder="Enter your diastole",
                                 help="The diastole of the blood pressure measurement of the person. On the development stage the minimum is 75 and the maximum is 95.",
                                 )
    heartRateInp = st.number_input("Heart Rate",
                                  min_value=65,
                                  max_value=86,
                                  step=1,
                                  value=None,
                                  placeholder="Enter your heart rate",
                                  help="The resting heart rate of the person in beats per minute. On the development stage the minimum is 65 and the maximum is 86.",
                                  )
    dailyStepsInp = st.number_input("Daily Steps",
                                   min_value=3000,
                                   max_value=10000,
                                   step=1,
                                   value=None,
                                   placeholder="Enter your daily steps",
                                   help="The number of steps the person takes per day. On the development stage the minimum is 3000 and the maximum is 10000.",
                                   )
    if st.button("Submit for prediction"):
        dictSubmit = {
            "Gender": genderInp,
            "Age": ageInp,
            "Occupation": occupationInp,
            "Sleep Duration": sleepDurationInp,
            "Quality of Sleep": qualityOfSleepInp,
            "Physical Activity Level": physicalActivityInp,
            "Stress Level": stressLevelInp,
            "BMI Category": BMICategoryInp,
            "Sistole": sistoleInp,
            "Diastole": diastoleInp,
            "Heart Rate": heartRateInp,
            "Daily Steps": dailyStepsInp
        }
        dfSubmit = pd.DataFrame(dictSubmit, index=[0])
        print(dfSubmit)
        result = caseBaseReasoning.newCase(dfSubmit,"Final")
        st.header("Prediction Result")
        st.write(result)




with tab2:
    st.header("Our Comparison Data")
    st.dataframe(caseBaseReasoning.getDF())

with tab3:
    st.header("All Data")
    testCSV = pd.read_csv("Sleep_health_and_lifestyle_dataset_Preproc.csv")
    st.dataframe(testCSV)

with tab4:
    st.header("Weight Setting")
    genderW = st.number_input(label='Gender Weight', value=config['Gender'], min_value=1.0, max_value=10.0, step=1.0)
    ageW = st.number_input(label='Age Weight', value=config['Age'], min_value=1.0, max_value=10.0, step=1.0)
    occupationW = st.number_input(label='Occupation Weight', value=config['Occupation'], min_value=1.0, max_value=10.0, step=1.0)
    sleepDurationW = st.number_input(label='Sleep Duration Weight', value=config['Sleep Duration'], min_value=1.0, max_value=10.0, step=1.0)
    qualityOfSleepW = st.number_input(label='Quality of Sleep Weight', value=config['Quality of Sleep'], min_value=1.0, max_value=10.0, step=1.0)
    physicalActivityLevelW = st.number_input(label='Physical Activity Level', value=config['Physical Activity Level'], min_value=1.0, max_value=10.0, step=1.0)
    stressLevelW = st.number_input(label='Stress Level', value=config['Stress Level'], min_value=1.0, max_value=10.0, step=1.0)
    BMICategoryW = st.number_input(label='BMI Category', value=config['BMI Category'], min_value=1.0, max_value=10.0, step=1.0)
    sistoleW = st.number_input(label='Sistole', value=config['Sistole'], min_value=1.0, max_value=10.0, step=1.0)
    diastoleW = st.number_input(label='Diastole', value=config['Diastole'], min_value=1.0, max_value=10.0, step=1.0)
    dailyStepsW = st.number_input(label='Daily Steps', value=config['Daily Steps'], min_value=1.0, max_value=10.0, step=1.0)
    st.header("Threshold Setting")
    thresholdInp = st.number_input(label='Threshold', value=config['Threshold'], min_value=0.1, max_value=1.0, step=0.05)

    if st.button("Update Setting"):
        try:
            newWeights = [
                float(genderW),
                float(ageW),
                float(occupationW),
                float(sleepDurationW),
                float(qualityOfSleepW),
                float(physicalActivityLevelW),
                float(stressLevelW),
                float(BMICategoryW),
                float(sistoleW),
                float(diastoleW),
                float(dailyStepsW)
            ]
            caseBaseReasoning, config = startCBR(newWeights, float(thresholdInp))
            st.write("Config Berhasil Diupdate")
        except ValueError as e:
            st.error(str(e))
            st.error("Pastikan semua input adalah angka (integer) yang valid.")
        except KeyError as e:
            st.error(f"Kunci konfigurasi tidak ditemukan: {e}. Harap periksa pengaturan Anda.")

