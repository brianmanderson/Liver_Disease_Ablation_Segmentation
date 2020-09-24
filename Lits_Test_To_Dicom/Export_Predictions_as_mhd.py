__author__ = 'Brian M Anderson'
# Created on 9/24/2020

from connect import *
import os

class Change_Patient(object):
    def __init__(self):
        self.patient_db = get_current('PatientDB')

    def process(self, MRN):
        self.MRN = MRN
        print('got here')
        info_all = self.patient_db.QueryPatientInfo(Filter={"PatientID": self.MRN}, UseIndexService=False)
        if not info_all:
            info_all = self.patient_db.QueryPatientInfo(Filter={"PatientID": self.MRN}, UseIndexService=True)
        for info in info_all:
            if info['PatientID'] == self.MRN:
                self.patient = self.patient_db.LoadPatient(PatientInfo=info, AllowPatientUpgrade=True)
                self.MRN = self.patient.PatientID
                return self.patient
        print('did not find a patient')
        return None

patient_holder = Change_Patient()
out_path_base = r'H:\Liver_Disease_Ablation\LiTs_Test\Nifti\predictions'
for pat_id in range(70):
    MRN = 'Lits_Test_{}'.format(pat_id)
    out_path_liver = os.path.join(out_path_base, 'Liver_{}.mhd'.format(pat_id))
    out_path_disease = os.path.join(out_path_base, 'Pred_{}.mhd'.format(pat_id))
    if os.path.exists(out_path_liver):
        continue
    try:
        patient = patient_holder.process(MRN)
    except:
        continue
    if patient is None:
        continue
    for case in patient.Cases:
        for exam in case.Examinations:
            continue #  Just take the last case and last exam
    rois_in_case = []
    for roi in case.PatientModel.RegionsOfInterest:
        rois_in_case.append(roi.Name)
    if 'Liver_BMA_Program_4' not in rois_in_case:
        has_liver = False
    else:
        has_liver = case.PatientModel.StructureSets[exam.Name].RoiGeometries['Liver_BMA_Program_4'].HasContours()
    if 'Liver_Disease_Ablation_BMA_Program_0' not in rois_in_case:
        has_disease = False
    else:
        has_disease = case.PatientModel.StructureSets[exam.Name].RoiGeometries['Liver_Disease_Ablation_BMA_Program_0'].HasContours()
    if has_liver:
        case.PatientModel.StructureSets[exam.Name].SimplifyContours(
            RoiNames=['Liver_BMA_Program_4'], RemoveHoles3D=False, RemoveSmallContours=False,
            ReduceMaxNumberOfPointsInContours=False, MaxNumberOfPoints=None,
            CreateCopyOfRoi=False, ResolveOverlappingContours=True)
    if has_disease:
        case.PatientModel.StructureSets[exam.Name].SimplifyContours(
            RoiNames=['Liver_Disease_Ablation_BMA_Program_0'], RemoveHoles3D=False, RemoveSmallContours=False,
            ReduceMaxNumberOfPointsInContours=False, MaxNumberOfPoints=None,
            CreateCopyOfRoi=False, ResolveOverlappingContours=True)
    patient.Save()
    if has_liver:
        case.PatientModel.StructureSets[exam.Name].RoiGeometries['Liver_BMA_Program_4'].ExportRoiGeometryAsMetaImage(
            MetaFileName=out_path_liver, AsExamination=True)
    if has_disease:
        case.PatientModel.StructureSets[exam.Name].RoiGeometries['Liver_Disease_Ablation_BMA_Program_0'].ExportRoiGeometryAsMetaImage(
            MetaFileName=out_path_disease, AsExamination=True)