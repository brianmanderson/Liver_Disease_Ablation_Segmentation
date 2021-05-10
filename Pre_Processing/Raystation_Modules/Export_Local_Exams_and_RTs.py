__author__ = 'Brian M Anderson'
# Created on 5/10/2021
import os
import pandas as pd
from connect import *


class ChangePatient(object):
    def __init__(self):
        self.patient_db = get_current("PatientDB")
        self.MRN = 0

    def change_patient(self, MRN):
        print('got here')
        self.MRN = MRN
        info_all = self.patient_db.QueryPatientInfo(Filter={"PatientID": self.MRN}, UseIndexService=False)
        if not info_all:
            info_all = self.patient_db.QueryPatientInfo(Filter={"PatientID": self.MRN}, UseIndexService=True)
        for info in info_all:
            if info['PatientID'] == self.MRN and info['LastName'].find('_') == -1:
                return self.patient_db.LoadPatient(PatientInfo=info, AllowPatientUpgrade=True)
        print('did not find a patient')
        return None


def export(case, exam_name, path):
    case.ScriptableDicomExport(ExportFolderPath=path, Examinations=[exam_name],
                               RtStructureSetsForExaminations=[exam_name])
    return None


def return_MRN_exam_dict(df):
    patient_dict = {}
    for index in df.index:
        MRN = df['MRN'][index]
        exam = df['Ablation_Exam'][index]
        if MRN not in patient_dict:
            patient_dict[MRN] = []
        if exam not in patient_dict[MRN]:
            patient_dict[MRN].append(exam)
    return patient_dict


def main():
    out_path = r'H:\Liver_Disease_Ablation\Dicom_Exports'
    patient_changer = ChangePatient()
    df = pd.read_excel(r'\\mymdafiles\di_data1\Morfeus\BMAnderson\Modular_Projects'
                       r'\Liver_Local_Recurrence_Work\Predicting_Recurrence\RetroAblation.xlsx', sheet_name='Refined')
    patient_dict = return_MRN_exam_dict(df=df)
    for MRN in patient_dict:
        try:
            patient = patient_changer.change_patient(MRN)
        except:
            continue
        if patient is not None:
            case = patient.Cases[0]
            for exam in patient_dict[MRN]:
                if exam in [e.Name for e in case.Examinations]:
                    exam_out_path = os.path.join(out_path, MRN, exam)
                    if not os.path.exists(exam_out_path):
                        os.makedirs(exam_out_path)
                    if not os.listdir(exam_out_path):
                        export(case=case, exam_name=exam, path=exam_out_path)
    return None


if __name__ == '__main__':
    main()
