__author__ = 'Brian M Anderson'
# Created on 6/9/2020

import os
from connect import *
import time, getpass


class create_RT_Structure():
    def __init__(self,roi_name):
        self.roi_name = roi_name
        self.version_name = '_BMA_Program_0'
        try:
            self.patient_db = get_current('PatientDB')
            self.patient = get_current('Patient')
            self.case = get_current('Case')
            self.MRN = self.patient.PatientID
        except:
            xxx = 1

    def ChangePatient(self, MRN):
        print('got here')
        self.MRN = MRN
        info_all = self.patient_db.QueryPatientInfo(Filter={"PatientID": self.MRN}, UseIndexService=False)
        if not info_all:
            info_all = self.patient_db.QueryPatientInfo(Filter={"PatientID": self.MRN}, UseIndexService=True)
        if len(info_all) > 1:
            print('{} has multiple ids'.format(MRN))
        for info in info_all:
            if info['PatientID'] == self.MRN and info['LastName'].find('_') == -1:
                self.patient = self.patient_db.LoadPatient(PatientInfo=info, AllowPatientUpgrade=True)
                self.MRN = self.patient.PatientID
                return None
        print('did not find a patient')

    def create_RT_Liver(self, exam):
        self.export(exam)
        if not self.has_contours:
            self.import_data(exam)
        else:
            print('Already has contours defined')

    def get_rois_in_case(self):
        self.rois_in_case = []
        for roi in self.case.PatientModel.RegionsOfInterest:
            self.rois_in_case.append(roi.Name)

    def export(self, exam):
        self.get_rois_in_case()
        roi_name = self.roi_name
        roi_name += '_Auto_Contour'
        self.MRN = self.patient.PatientID
        self.base_path = '\\\\mymdafiles\\ou-radonc\\Raystation\\Clinical\\Auto_Contour_Sites\\'
        #if not check_any_contours(case,exam): Doesn't work the way I want it to
        self.path = os.path.join(self.base_path,roi_name,'Input_3',self.MRN)
        self.patient.Save()
        actual_roi_names = [self.roi_name + self.version_name]
        self.has_contours = True
        for actual_roi_name in actual_roi_names:
            if actual_roi_name in self.rois_in_case:
                if not self.case.PatientModel.StructureSets[exam.Name].RoiGeometries[actual_roi_name].HasContours():
                    self.has_contours = False
                    break
            else:
                self.has_contours = False
                break
        has_liver = False
        for actual_roi_name in ['Liver','Liver_BMA_Program_4']:
            if actual_roi_name in self.rois_in_case:
                if self.case.PatientModel.StructureSets[exam.Name].RoiGeometries[actual_roi_name].HasContours():
                    has_liver = True
                    break
        if not has_liver:
            print('You need a contour named Liver or Liver_BMA_Program_4')
            self.has_contours = True
        if self.has_contours:
            return None
        self.patient.Save()
        self.Export_Dicom(exam,self.path)

    def import_data(self, exam):
        roi_name = self.roi_name
        actual_roi_name = roi_name + self.version_name
        roi_name += '_Auto_Contour'
        if actual_roi_name in self.rois_in_case:
            if self.case.PatientModel.StructureSets[exam.Name].RoiGeometries[actual_roi_name].HasContours():
                return None # Already have the contours for this patient
        data = exam.GetAcquisitionDataFromDicom()
        SeriesUID = data['SeriesModule']['SeriesInstanceUID']
        output_path = os.path.join(self.base_path,roi_name,'Output',self.MRN,SeriesUID)
        self.cleanout_folder(output_path)
        print('Now waiting for RS to be made')
        self.import_RT = False
        self.check_folder(output_path)
        print('Import RT structure!')
        if self.import_RT:
            self.importRT(output_path)
            self.get_rois_in_case()
            if actual_roi_name in self.rois_in_case:
                if self.case.PatientModel.StructureSets[exam.Name].RoiGeometries[actual_roi_name].HasContours():
                    self.simplify_contours(exam,actual_roi_name)
        self.cleanout_folder(output_path)
        return None

    def simplify_contours(self, exam, roi_name):
        self.case.PatientModel.StructureSets[exam.Name].SimplifyContours(
            RoiNames=[roi_name], RemoveHoles3D=False, RemoveSmallContours=True,
            AreaThreshold=0.5, ReduceMaxNumberOfPointsInContours=False, MaxNumberOfPoints=None,
            CreateCopyOfRoi=False, ResolveOverlappingContours=False)
        self.case.PatientModel.StructureSets[exam.Name].SimplifyContours(
            RoiNames=[roi_name], RemoveHoles3D=True, RemoveSmallContours=False,
            ReduceMaxNumberOfPointsInContours=False, MaxNumberOfPoints=None,
            CreateCopyOfRoi=False, ResolveOverlappingContours=True)
        self.patient.Save()

    def Export_Dicom(self,exam, path):
        data = exam.GetAcquisitionDataFromDicom()
        SeriesUID = data['SeriesModule']['SeriesInstanceUID']
        export_path = os.path.join(path,SeriesUID)
        if not os.path.exists(export_path):
            print('making path')
            os.makedirs(export_path)
        print(export_path)
        if not os.path.exists(os.path.join(export_path,'Completed.txt')):
            self.case.ScriptableDicomExport(ExportFolderPath=export_path, Examinations=[exam.Name],
                                            RtStructureSetsForExaminations=[exam.Name])
            fid = open(os.path.join(export_path,'Completed.txt'),'w+')
            fid.close()
        return None

    def check_folder(self,output_path):
        print(output_path)
        while not os.path.exists(output_path):
            time.sleep(1)
        print('path exists, waiting for file')
        while not os.path.exists(os.path.join(output_path,'Completed.txt')) and not os.path.exists(os.path.join(output_path,'Failed.txt')):
            time.sleep(1)
        if os.path.exists(os.path.join(output_path,'Completed.txt')):
            self.import_RT = True
        return None

    def importRT(self,file_path):
        try:
            self.patient.ImportDicomDataFromPath(Path=file_path,CaseName=self.case.CaseName,SeriesFilter={},ImportFilters=[])
        except:
            pi = self.patient_db.QueryPatientsFromPath(Path=file_path, SearchCriterias={'PatientID': self.MRN})[0]
            studies = self.patient_db.QueryStudiesFromPath(Path=file_path,
                                                           SearchCriterias=pi)
            series = []
            for study in studies:
                series += self.patient_db.QuerySeriesFromPath(Path=file_path,
                                                              SearchCriterias=study)
            self.patient.ImportDataFromPath(Path=file_path, CaseName=self.case.CaseName,
                                            SeriesOrInstances=series, AllowMismatchingPatientID=True)
        return None

    def cleanout_folder(self,dicom_dir):
        print('Cleaning up: Removing imported DICOMs, please check output folder for result')
        if os.path.exists(dicom_dir):
            files = os.listdir(dicom_dir)
            for file in files:
                if file.find('user_') != 0:
                    os.remove(os.path.join(dicom_dir,file))
            un = getpass.getuser()
            fid = open(os.path.join(dicom_dir,'user_{}.txt'.format(un)),'w+')
            fid.close()
        return None


def create_RT_of_disease_ablation(prediction_class):
    text_file = r'H:\Liver_Disease_Ablation\Raystation_Disease_Status'
    csv_path = r'H:\Modular_Projects\Liver_Disease_Ablation_Segmentation\Raystation_Code\MRNs_All_Primary_Secondary_exam.csv'
    fid = open(csv_path)
    fid.readline()
    data = []
    for line in fid:
        line = line.strip('\n')
        data.append(line.split(','))
    fid.close()
    for index in range(len(data)):
        MRN, primary, secondary = data[index]
        status_file = os.path.join(text_file, '{}_Predicted.txt'.format(MRN))
        if os.path.exists(status_file):
            continue
        try:
            prediction_class.ChangePatient(MRN)
        except:
            continue
        for case in prediction_class.patient.Cases:
            prediction_class.case = case
            # prediction_class.get_rois_in_case()
            # if 'Liver_Disease_Ablation_BMA_Program_0' in prediction_class.rois_in_case:
            #     prediction_class.case.PatientModel.RegionsOfInterest['Liver_Disease_Ablation_BMA_Program_0'].DeleteRoi()
            #     prediction_class.patient.Save()
            # case.SetCurrent()
            # continue
            for exam in [primary, secondary]:
                try:
                    prediction_class.create_RT_Liver(case.Examinations[exam])
                except:
                    xxx = 1
            fid = open(status_file,'w+')
            fid.close()


def export_roi(case, exam_name, name, roi_name, export_path):
    case.PatientModel.StructureSets[exam_name].RoiGeometries[roi_name].ExportRoiGeometryAsMetaImage(
        MetaFileName=export_path.format('{}_{}'.format(roi_name,name)), AsExamination=True)
    return None


def export_RTs_as_mhds(prediction_class):
    text_file = r'H:\Liver_Disease_Ablation\Raystation_Disease_Export_Status'
    csv_path = r'H:\Modular_Projects\Liver_Disease_Ablation_Segmentation\Raystation_Code\MRNs_All_Primary_Secondary_exam.csv'
    fid = open(csv_path)
    fid.readline()
    data = []
    for line in fid:
        line = line.strip('\n')
        data.append(line.split(','))
    fid.close()
    pred_roi = 'Liver_Disease_Ablation_BMA_Program_0'
    base_export = r'H:\Liver_Disease_Ablation\Disease_Ablation_From_Raystation_Test'
    for index in range(len(data)):
        MRN, primary, secondary = data[index]
        patient_path = os.path.join(base_export,MRN)
        if os.path.exists(patient_path):
            continue
        status_file = os.path.join(text_file, '{}_Predicted.txt'.format(MRN))
        if os.path.exists(status_file):
            continue
        try:
            prediction_class.ChangePatient(MRN)
        except:
            continue
        for case in prediction_class.patient.Cases:
            prediction_class.case = case
            rois_in_case = []
            for roi in case.PatientModel.RegionsOfInterest:
                rois_in_case.append(roi.Name)
            break_out = False
            for roi in ['Liver', pred_roi, 'GTV', 'Ablation']:
                if roi not in rois_in_case:
                    break_out = True
                    break
            if break_out:
                continue
            os.makedirs(patient_path)
            export_path = os.path.join(patient_path,'{}.mhd')
            for exam, roi_name, name in zip([primary, secondary],['GTV','Ablation'],['Primary','Secondary']):
                if case.PatientModel.StructureSets[exam].RoiGeometries[roi_name].HasContours() and case.PatientModel.StructureSets[primary].RoiGeometries[pred_roi].HasContours():
                    export_roi(case=case, exam_name=exam, name=name, roi_name=roi_name, export_path=export_path)
                    export_roi(case=case, exam_name=exam, name=name, roi_name=pred_roi, export_path=export_path)
                    if case.PatientModel.StructureSets[exam].RoiGeometries['Liver'].HasContours():
                        export_roi(case=case, exam_name=exam, name=name, roi_name='Liver', export_path=export_path)
                    elif case.PatientModel.StructureSets[exam].RoiGeometries['Liver_BMA_Program_4'].HasContours():
                        export_roi(case=case, exam_name=exam, name=name, roi_name='Liver_BMA_Program_4', export_path=export_path)
    return None


def main():
    create_disease = False
    prediction_class = create_RT_Structure(roi_name='Liver_Disease_Ablation')
    if create_disease:
        create_RT_of_disease_ablation(prediction_class)

    export_RTs = True
    if export_RTs:
        export_RTs_as_mhds(prediction_class)


if __name__ == '__main__':
    main()
