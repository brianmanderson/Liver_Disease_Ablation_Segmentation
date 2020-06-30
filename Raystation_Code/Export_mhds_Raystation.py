__author__ = 'Brian M Anderson'
# Created on 6/26/2020

from connect import *
import os, time


path = r'H:\Single_Site'
for i in os.listdir(path):
    os.remove(os.path.join(path, i))
patient_db = get_current('PatientDB')
patient = get_current('Patient')
case = get_current('Case')
exam = get_current('Examination')
rois_in_case = []
for roi in case.PatientModel.RegionsOfInterest:
    rois_in_case.append(roi.Name)

exported = False
for roi in rois_in_case:
    if roi in ['Ablation','GTV','Liver_Disease_Ablation_BMA_Program_0']:
        if case.PatientModel.StructureSets[exam.Name].RoiGeometries[roi].HasContours():
            case.PatientModel.StructureSets[exam.Name].RoiGeometries[roi].ExportRoiGeometryAsMetaImage(
                MetaFileName=os.path.join(path,'{}.mhd'.format(roi)), AsExamination=True)
            exported = True

if exported:
    case.ScriptableDicomExport(ExportFolderPath=path, Examinations=[exam.Name], RtStructureSetsForExaminations=[])
    fid = open(os.path.join(path,'Completed_Export.txt'),'w+')
    fid.close()
    while not os.path.exists(os.path.join(path, 'Completed.txt')):
        print('Waiting...')
        time.sleep(5)
    pi = patient_db.QueryPatientsFromPath(Path=path, SearchCriterias={'PatientID': patient.PatientID})[0]
    studies = patient_db.QueryStudiesFromPath(Path=path, SearchCriterias=pi)
    series = []
    for study in studies:
        series += patient_db.QuerySeriesFromPath(Path=path, SearchCriterias=study)
    patient.ImportDataFromPath(Path=path, CaseName=case.CaseName, SeriesOrInstances=series, AllowMismatchingPatientID=True)
    for i in os.listdir(path):
        os.remove(os.path.join(path,i))