__author__ = 'Brian M Anderson'
# Created on 4/4/2020
'''
This module is to mask the disease contours to be within the liver and fill any holes
'''

from connect import *


class Create_Smoothed_ROI(object):
    def __init__(self):
        self.patient_db = get_current('PatientDB')
        self.exam = None
        self.case = None

    def ChangePatient(self, MRN):
        print('got here')
        self.MRN = MRN
        info_all = self.patient_db.QueryPatientInfo(Filter={"PatientID": self.MRN}, UseIndexService=False)
        if not info_all:
            info_all = self.patient_db.QueryPatientInfo(Filter={"PatientID": self.MRN}, UseIndexService=True)
        for info in info_all:
            if info['PatientID'] == self.MRN:
                self.patient = self.patient_db.LoadPatient(PatientInfo=info, AllowPatientUpgrade=True)
                self.MRN = self.patient.PatientID
                return None

    def get_rois_in_case(self):
        self.rois_in_case = [roi.Name for roi in self.case.PatientModel.RegionsOfInterest]

    def copy_and_smooth_contour(self,region_roi, expan=0.5, new_roi_name='Ablation_BMA'):
        if new_roi_name not in self.rois_in_case:
            self.case.PatientModel.CreateRoi(Name=new_roi_name,Color='Yellow')
        with CompositeAction('ROI Algebra (new_ablation, Image set: ' + self.exam.Name + ')'):
            self.case.PatientModel.RegionsOfInterest[new_roi_name].CreateAlgebraGeometry(Examination=self.exam,
                                                                                      Algorithm="Auto",
                                                                                      ExpressionA={'Operation': "Union",
                                                                                                   'SourceRoiNames': [
                                                                                                       region_roi],
                                                                                                   'MarginSettings': {
                                                                                                       'Type': "Expand",
                                                                                                       'Superior': expan,
                                                                                                       'Inferior': expan,
                                                                                                       'Anterior': expan,
                                                                                                       'Posterior': expan,
                                                                                                       'Right': expan,
                                                                                                       'Left': expan}},
                                                                                      ExpressionB={'Operation': "Union",
                                                                                                   'SourceRoiNames': [],
                                                                                                   'MarginSettings': {
                                                                                                       'Type': "Expand",
                                                                                                       'Superior': 0,
                                                                                                       'Inferior': 0,
                                                                                                       'Anterior': 0,
                                                                                                       'Posterior': 0,
                                                                                                       'Right': 0,
                                                                                                       'Left': 0}},
                                                                                      ResultOperation="None",
                                                                                      ResultMarginSettings={
                                                                                          'Type': "Contract",
                                                                                          'Superior': expan,
                                                                                          'Inferior': expan,
                                                                                          'Anterior': expan,
                                                                                          'Posterior': expan,
                                                                                          'Right': expan, 'Left': expan})

    def resolve_overlapping_contours(self, roi_name):
        self.case.PatientModel.StructureSets[self.exam.Name].SimplifyContours(RoiNames=[roi_name],
                                                                              ResolveOverlappingContours=True)

    def simplify_contours(self, area_threshold=0.1, roi_name='Ablation_BMA'):
        self.case.PatientModel.StructureSets[self.exam.Name].SimplifyContours(RoiNames=[roi_name], RemoveHoles3D=True,
                                                                           RemoveSmallContours=True, AreaThreshold=area_threshold,
                                                                           ReduceMaxNumberOfPointsInContours=False,
                                                                           MaxNumberOfPoints=None,
                                                                           CreateCopyOfRoi=False,
                                                                           ResolveOverlappingContours=False)

    def bring_within_liver(self,liver_roi, ablation_roi='Ablation_BMA'):

        with CompositeAction('ROI Algebra (new_ablation, Image set: ' + self.exam.Name + ')'):
            self.case.PatientModel.RegionsOfInterest[ablation_roi].CreateAlgebraGeometry(Examination=self.exam,
                                                                                      Algorithm="Auto",
                                                                                      ExpressionA={'Operation': "Union",
                                                                                                   'SourceRoiNames': [
                                                                                                       ablation_roi],
                                                                                                   'MarginSettings': {
                                                                                                       'Type': "Expand",
                                                                                                       'Superior': 0,
                                                                                                       'Inferior': 0,
                                                                                                       'Anterior': 0,
                                                                                                       'Posterior': 0,
                                                                                                       'Right': 0,
                                                                                                       'Left': 0}},
                                                                                      ExpressionB={'Operation': "Union",
                                                                                                   'SourceRoiNames': [
                                                                                                       liver_roi],
                                                                                                   'MarginSettings': {
                                                                                                       'Type': "Expand",
                                                                                                       'Superior': 0,
                                                                                                       'Inferior': 0,
                                                                                                       'Anterior': 0,
                                                                                                       'Posterior': 0,
                                                                                                       'Right': 0,
                                                                                                       'Left': 0}},
                                                                                      ResultOperation="Intersection",
                                                                                      ResultMarginSettings={
                                                                                          'Type': "Expand",
                                                                                          'Superior': 0, 'Inferior': 0,
                                                                                          'Anterior': 0, 'Posterior': 0,
                                                                                          'Right': 0,
                                                                                          'Left': 0})


def ChangePatient(patient_db, patient_id):
    info_all = patient_db.QueryPatientInfo(Filter={"PatientID": patient_id})
    # If it isn't, see if it's in the secondary database
    if not info_all:
        info_all = patient_db.QueryPatientInfo(Filter={"PatientID": patient_id}, UseIndexService=False)
    info = []
    for info_temp in info_all:
        if info_temp['PatientID'] == patient_id:
            info = info_temp
            break
    return patient_db.LoadPatient(PatientInfo=info, AllowPatientUpgrade=False)


def main():
    create_smooth_roi_class = Create_Smoothed_ROI()
    patient_db = get_current("PatientDB")
    new_roi = 'Disease_BMA_0'
    base_roi = 'Disease_BMA'
    info_all = patient_db.QueryPatientInfo(Filter={"PatientID": 'LiTs'})
    for info in info_all:
        patient = patient_db.LoadPatient(PatientInfo=info, AllowPatientUpgrade=False)
        create_smooth_roi_class.patient = patient
        for case in patient.Cases:
            # case.SetCurrent()
            create_smooth_roi_class.case = case
            rois_in_case = []
            for roi in case.PatientModel.RegionsOfInterest:
                rois_in_case.append(roi.Name)
            create_smooth_roi_class.rois_in_case = rois_in_case
            if base_roi not in rois_in_case or new_roi in rois_in_case:
                continue
            for exam in case.Examinations:
                if case.PatientModel.StructureSets[exam.Name].RoiGeometries[base_roi].HasContours():
                    create_smooth_roi_class.exam = exam
                    create_smooth_roi_class.copy_and_smooth_contour(base_roi, expan=0, new_roi_name=new_roi)
                    create_smooth_roi_class.resolve_overlapping_contours(new_roi)
                    create_smooth_roi_class.simplify_contours(0, roi_name=new_roi)
                    patient.Save()


if __name__ == '__main__':
    main()
