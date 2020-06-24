__author__ = 'Brian M Anderson'
# Created on 4/5/2020

from connect import *
import os


def export_roi(patient, case, exam, roi, base_export_path=''):
    MRN = patient.PatientID
    out_path = os.path.join(base_export_path,'{}_{}.mhd'.format(MRN,roi))
    if os.path.exists(out_path):
        return None
    if not os.path.exists(base_export_path):
        os.makedirs(base_export_path)
    case.PatientModel.StructureSets[exam.Name].RoiGeometries[roi].ExportRoiGeometryAsMetaImage(MetaFileName=out_path,
                                                                                               AsExamination=True)
    return None


def main():
    patient_db = get_current("PatientDB")
    base_export_path = 'H:\Liver_Disease_Ablation\Raystation_Exports'
    info_all = patient_db.QueryPatientInfo(Filter={"PatientID": 'LiTs'})
    new_roi = 'Disease_BMA_0'
    for info in info_all:
        patient = patient_db.LoadPatient(PatientInfo=info, AllowPatientUpgrade=False)
        for case in patient.Cases:
            # case.SetCurrent()
            rois_in_case = []
            for roi in case.PatientModel.RegionsOfInterest:
                rois_in_case.append(roi.Name)
            if new_roi not in rois_in_case:
                continue
            for exam in case.Examinations:
                if case.PatientModel.StructureSets[exam.Name].RoiGeometries[new_roi].HasContours():
                    export_roi(patient, case, exam, roi=new_roi, base_export_path=base_export_path)


if __name__ == '__main__':
    main()
