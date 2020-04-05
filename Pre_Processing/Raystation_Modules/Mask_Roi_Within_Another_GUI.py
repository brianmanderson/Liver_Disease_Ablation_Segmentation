__author__ = 'Brian M Anderson'
# Created on 4/4/2020

import clr, sys

clr.AddReference('System')
import wpf

clr.AddReference("System.Xml")

from System import (IO, Windows)
from System.Windows import MessageBox, Application, Window
from System.Xml import XmlReader
from System.IO import StringReader

from connect import *

#####################
# xaml code for GUI
#
xaml = """
<Window xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
        xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
        Title="CBCT conversion setup"
        Margin="10,10,10,10"
        SizeToContent="WidthAndHeight"
        Top="200"
        WindowStartupLocation="Manual">
    <Grid>
        <Grid.ColumnDefinitions>
            <ColumnDefinition Width="*" />
            <ColumnDefinition Width="*" />
        </Grid.ColumnDefinitions>
        <Grid.RowDefinitions>
            <RowDefinition Height="*" />
            <RowDefinition Height="*" />
            <RowDefinition Height="*" />
            <RowDefinition Height="*" />
            <RowDefinition Height="*" />
        </Grid.RowDefinitions>
        <TextBlock Grid.Row="0"
                   Grid.Column="0"
                   Margin="5"
                   Width="200"
                   Text="Select liver roi:" />
        <ComboBox Name="LiverROI"
                  Grid.Row="0"
                   Width="200"
                  Grid.Column="1"
                  Margin="5" />
        <TextBlock Grid.Row="1"
                   Grid.Column="0"
                   Margin="5"
                   Width="200"
                   Text="Select region grown roi:" />
        <ComboBox Name="RegionGrownROI"
                  Grid.Row="1"
                   Width="200"
                  Grid.Column="1"
                  Margin="5" />
                <Button Grid.Row="4" Grid.Column="0" Grid.ColumnSpan="2" HorizontalAlignment="Right"
                Margin="5" Width="80" Content="Compute" Name="ButtonCompute"/>
    </Grid>
</Window>
"""

#
# END xaml-code
#####################


xr = XmlReader.Create(StringReader(xaml))

error_message_header = "Missing information / setup"


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
            self.case.PatientModel.CreateRoi(Name=new_roi_name,Color='Red')
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


class SetupDialog(Window):


    def compute_clicked(self, sender, event):
        liver_roi = self.LiverROI.SelectedItem
        region_roi = self.RegionGrownROI.SelectedItem
        # self.create_smooth_roi_class.run_whole_thing(self.exam.Name)

        self.create_smooth_roi_class.copy_and_smooth_contour(region_roi, expan=0)

        if liver_roi != 'None':
            self.create_smooth_roi_class.bring_within_liver(liver_roi)
        # self.create_smooth_roi_class.simplify_contours()
        self.patient.Save()
        self.window.Close()

    def __init__(self):
        self.window = wpf.LoadComponent(self, xr)
        self.patient = get_current('Patient')
        self.patient_db = get_current("PatientDB")
        self.case = get_current('Case')
        self.exam = get_current('Examination')

        self.create_smooth_roi_class = Create_Smoothed_ROI()
        self.create_smooth_roi_class.patient = self.patient
        self.create_smooth_roi_class.exam = self.exam
        self.create_smooth_roi_class.case = self.case
        self.create_smooth_roi_class.get_rois_in_case()

        self.rois_in_case = ['None'] + self.create_smooth_roi_class.rois_in_case

        self.LiverROI.ItemsSource = self.rois_in_case
        self.LiverROI.SelectedIndex = 0
        self.RegionGrownROI.ItemsSource = self.rois_in_case
        self.RegionGrownROI.SelectedIndex = 0

        self.ButtonCompute.Click += self.compute_clicked


####################################
#     MAIN PROGRAM
#

print("Program started...")
#
# Check that a patient is open
# In case no, the script will terminate
#
patient = None
try:
    patient = get_current("Patient")
except Exception as e:
    MessageBox.Show("No patient loaded.", error_message_header)
    sys.exit()

#
# Check that a case is loaded
# In case no, the script will terminate
#
case = None
try:
    case = get_current("Case")
except Exception as e:
    MessageBox.Show("No case active.", error_message_header)
    sys.exit()

Application().Run(SetupDialog())

print("...program ended")

