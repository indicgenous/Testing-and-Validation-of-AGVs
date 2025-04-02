import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import wilcoxon
# from sklearn.linear_model import LinearRegression
import os


class vru_test:
    def __init__(self, dataset):
      if isinstance(dataset, str):
          try:
              self.df = pd.read_csv(
                  dataset,
                  delimiter=';',
                  header=122,
                  on_bad_lines='skip'
              )
              # Strip any extra spaces in column names for consistency
              self.df.columns = self.df.columns.str.strip()
              print("Columns in dataset:", self.df.columns)  # Print available columns for debugging
          except FileNotFoundError:
              raise Exception(f"File not found: {dataset}")
      elif isinstance(dataset, pd.DataFrame):  # Check if dataset is a DataFrame
            self.df = dataset.copy()  # Use the provided DataFrame directly
            print("Using provided DataFrame.")
      else:
          raise TypeError("Invalid dataset type. Expected str (file path) or pd.DataFrame.")

      # Ensure all required columns are present in the dataset
      self.validate_columns()

      # Process the dataset to compute new attributes
      self.process_dataset() 
      self.all_plot()
      self.ttc_plot()
      self.bd_plot()
      self.find_peak()


      # The function definations start here ------------------------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    def validate_columns(self):
        """
        Validate that all required columns are present in the dataset.
        If any required columns are missing, raise a KeyError.
        """
        required_columns = [
            'Lat[wgs84].1', 'Lng[wgs84].1', 'vX[m/s]', 'vY[m/s]',
            'Acceleration[m/s²].1', 'Lat[wgs84].2', 'Lng[wgs84].2',
            'vX[m/s].1', 'vY[m/s].1', 'Acceleration[m/s²].2', 'Timestamp[ms]'
        ]

        # Identify missing columns
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        
        if missing_columns:
            raise KeyError(f"Missing columns in dataset: {missing_columns}")

    def longitude_to_meters(self, longitudes, latitudes):
        """
        Convert longitude values to distances in meters based on latitude.
        This accounts for Earth's curvature.
        """
        latitudes_radians = np.radians(latitudes)  # Convert latitude to radians
        return longitudes * 111320 * np.cos(latitudes_radians)  # Convert to meters

    def process_dataset(self):
        """
        Process the dataset to calculate new attributes such as:
        - Longitudinal distance
        - Relative velocity
        - Relative acceleration
        """
        # Extract pedestrian coordinates
        self.ped_lat = self.df['Lat[wgs84].1']  # Pedestrian latitude
        self.ped_lng = self.df['Lng[wgs84].1']  # Pedestrian longitude

        # Extract pedestrian velocities
        self.ped_vel_x = self.df['vX[m/s]']  # Pedestrian velocity in x-direction
        self.ped_vel_y = self.df['vY[m/s]']  # Pedestrian velocity in y-direction

        # Extract pedestrian acceleration
        self.ped_acc = self.df['Acceleration[m/s²].1']  # Pedestrian acceleration

        # Extract vehicle coordinates
        self.vcl_lat = self.df['Lat[wgs84].2']  # Vehicle latitude
        self.vcl_lng = self.df['Lng[wgs84].2']  # Vehicle longitude

        # Extract vehicle velocities
        self.vcl_vel_x = self.df['vX[m/s].1']  # Vehicle velocity in x-direction
        self.vcl_vel_y = self.df['vY[m/s].1']  # Vehicle velocity in y-direction

        # Extract vehicle acceleration
        self.vcl_acc = self.df['Acceleration[m/s²].2']  # Vehicle acceleration

        # Convert vehicle and pedestrian longitudes to distances in meters
        self.vcl_lng_m = self.longitude_to_meters(self.vcl_lng, self.vcl_lat)
        self.ped_lng_m = self.longitude_to_meters(self.ped_lng, self.ped_lat)

        # Compute the longitudinal distance between the vehicle and pedestrian
        self.df['del_s'] = self.vcl_lng_m - self.ped_lng_m

        # Compute the resultant velocity for both the vehicle and pedestrian
        self.df['rel_v'] = np.sqrt(
            (self.vcl_vel_x - self.ped_vel_x)**2 +
            (self.vcl_vel_y - self.ped_vel_y)**2
        )

        # Compute the relative acceleration between the vehicle and pedestrian
        self.df['rel_acc'] = self.vcl_acc - self.ped_acc
        return self.df['rel_v']



    def vel_plot(self):
        """
        Plot the relative velocity (`rel_v`) over time (`Timestamp[ms]`).
        This visualizes the changes in relative velocity between the vehicle and pedestrian.
        """
        plt.figure(figsize=(5,5))  # Set plot size
        plt.scatter(x=self.df['Timestamp[ms]'], y=self.df['rel_v'], s=1, alpha=0.8)
        plt.xlim(0, 15000)  # Set x-axis limits
        plt.gca().spines[['top', 'right']].set_visible(False)  # Hide top and right spines
        plt.xlabel('Timestamp [ms]')  # Label x-axis
        plt.ylabel('Relative Velocity (rel_v)')  # Label y-axis
        plt.title('Relative Velocity vs Time')  # Add plot title
        # plt.show()

    def s_plot(self):
        self.df.plot(x='Timestamp[ms]',y='del_s')
        plt.xlabel('Timestamp[ms]')
        plt.ylabel('Longitudinal distance between Vehical and Target')
        # plt.gca().spines[['top', 'right',]].set_visible(False)
        plt.title('Longitudinal distance between Vehical and Target')


    def vcl_acc_plot(self):
      x=self.df['Timestamp[ms]']
      y=self.vcl_acc
      plt.plot(x,y)
      plt.xlabel('Timestamp[ms]')
      plt.ylabel('vcl_acc')
      # plt.show()
      # plt.gca().spines[['top', 'right',]].set_visible(False)





    def all_plot(self,save_path="E:\TiHAN\Data_Pipeline/plots"):
        fig , axes = plt.subplots(2,2,figsize=(10,10))
        os.makedirs(save_path, exist_ok=True)

        plt.sca(axes[0,0])
        plt.plot(self.df['Timestamp[ms]'],self.df['del_s'])
        plt.xlabel('Timestamp[ms]')
        plt.ylabel('Longitudinal distance between Vehical and Target')
        # plt.gca().spines[['top', 'right',]].set_visible(False)
        axes[0,0].set_title('Longitudinal Distance')

        plt.sca(axes[0,1])
        plt.scatter(x=self.df['Timestamp[ms]'], y=self.df['rel_v'], s=1, alpha=0.8)
        plt.xlim(0, 15000)  # Set x-axis limits
        plt.gca().spines[['top', 'right']].set_visible(False)  # Hide top and right spines
        plt.xlabel('Timestamp [ms]')  # Label x-axis
        plt.ylabel('Relative Velocity (rel_v)')  # Label y-axis
        axes[0,1].set_title('Relative Velocity vs TimeStamp')

        plt.sca(axes[1,0])
        self.vcl_acc_plot()
        axes[1,0].set_title('vcl_acc vs TimeStamp')

        plt.sca(  axes[1,1])
        plt.scatter(self.df['Timestamp[ms]'],self.df['rel_acc'])
        plt.xlabel('Timestamp[ms]')
        plt.ylabel('rel_v')
        axes[1,1].set_title('rel_acc vs TimeStamp')
        plt.gca().spines[['top', 'right']].set_visible(False)
        # plt.xlim(0,10000)

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "all_plots.png"))
        print("All plot saved")


    def poly_reg(self):
      x = self.df['Timestamp[ms]'].values.reshape(-1, 1)
      y = self.df['rel_acc'].values
      degree =2
      mask = np.isnan(y)
      y=y[~mask]
      x=x[~mask]
      poly_features = PolynomialFeatures(degree=degree)
      x_poly = poly_features.fit_transform(x)
      model = LinearRegression()
      model.fit(x_poly, y)

      x_reg = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
      x_reg_poly = poly_features.transform(x_reg)
      y_reg = model.predict(x_reg_poly)


      plt.scatter(x, y, s=5, alpha=.8)
      plt.plot(x_reg, y_reg, color='red')
      # plt.xlim(0, 10000)
      plt.gca().spines[['top', 'right']].set_visible(False)
      plt.xlabel('Timestamp[ms]')
      plt.ylabel('rel_acc')
    #   plt.show()

    # TTC CALCULATION----------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>------------------>>>>>>>>>>>>>>>>>


    def ttc_s_plot(self,ttc):
        plt.scatter(self.df['del_s'],ttc,c='b')
        plt.xlabel('longitudinal distance')
        plt.ylabel('TTC')
        plt.ylim(0,5)
        plt.axhline(y=4,color='r',linestyle='--',label='Max TTC')
        plt.title('TTC vs. Longitudinal Distance')
        plt.grid(True)
        plt.legend()
        plt.gca().spines[['top', 'right',]].set_visible(False)
        # plt.show()

    def ttc_t_plot(self,ttc):
      # TTC calculation
      # Scatter plot to figure out the outliers
      from scipy.stats import zscore

      # Calculate z-scores for 'Timestamp[ms]'
      z_scores_x = zscore(self.df['Timestamp[ms]'])

      # Filter out NaN values from 'ttc' before calculating z-scores
      valid_ttc_indices = ~np.isnan(self.ttc)
      z_scores_y = zscore(self.ttc[valid_ttc_indices])

      # Align z_scores_x with valid_ttc_indices
      z_scores_x = z_scores_x[valid_ttc_indices]

      threshold = 2

      # Perform the logical OR operation on the aligned arrays
      outliers = (np.abs(z_scores_x) > threshold) | (np.abs(z_scores_y) > threshold)

      colors = ['red' if outlier else 'blue' for outlier in outliers]  # No need to flatten
      plt.scatter(self.df['Timestamp[ms]'][valid_ttc_indices], self.ttc[valid_ttc_indices], c=colors)
      # plt.xlim(0,10000)
      plt.ylim(0,5)
      plt.xlabel('Timestamp[ms]')
      plt.ylabel('TTC')
      plt.gca().spines[['top', 'right']].set_visible(False)
      # plt.show()


    def ttc_vplot(self,ttc):
      # Create a dummy category for the violin plot (if needed)
      # This allows you to make a single violin plot if you don't have any category in the dataset.
      self.df['Category'] = 'TTC'

      # Create the violin plot
      sns.violinplot(x='Category', y=ttc, data=self.df)
      plt.title('Violin Plot of TTC')



    def b_plot(self,ttc):
      sns.boxplot(self.ttc)
      plt.ylabel('All Timestamps')
      plt.xlabel('TTC')
      # plt.show()


    def ttc_calc(self):
      #initialization of the ttc vector with 0s
      self.ttc = np.zeros_like(self.df['rel_v'])

      #ttc calculation formula
      self.ttc = self.df['del_s'] / self.df['rel_v']

      # print(f"Total number of null values in TTC array :{np.isnan(self.ttc).sum()}")
      # ttc.dropna(inplace=True)

      return self.ttc

    def ttc_plot(self,save_path="E:\TiHAN\Data_Pipeline\plots"):
        self.ttc_calc()
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        os.makedirs(save_path, exist_ok=True)

        plt.sca(axes[0,0])
        self.ttc_t_plot(self.ttc)
        axes[0,0].set_title("TTC against Time_stamp")


        plt.sca(axes[0,1])
        self.ttc_s_plot(self.ttc)
        axes[0,1].set_title("TTC vs Longitudinal Distance")

        plt.sca(axes[1,0])
        self.b_plot(self.ttc)
        # plt.xlabel('TTC')
        # plt.ylabel('TTC')
        axes[1,0].set_title("BOXPLOT of TTC")

        plt.sca(axes[1,1])
        self.ttc_vplot(self.ttc)
        axes[1,1].set_ylabel("TTC against all TimeStamp")
        plt.savefig(os.path.join(save_path, "TTC_plots.png"))
        print("TTC plot saved")


        # Adjust layout
        # plt.tight_layout()
        # plt.show()

#  Braking demand calculation----------------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    def bd(self):
      self.bd = (self.df['rel_v'])**2 / (2 * self.df['del_s'])
      # self.bd = self.bd[:len(self.df['Timestamp[ms]'])]
      return self.bd



    # def bd_plot(self):
    #   # self.bd()
    #   self.bd = self.bd[:len(self.df['Timestamp[ms]'])]  # Truncate bd to match the length of df['Timestamp[ms]']
    #   fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    #   plt.sca(axes[0,0])
    #   plt.plot(self.df['Timestamp[ms]'], self.bd)
    #   plt.xlabel('Timestamp[ms]')
    #   plt.ylabel('Breaking Demand')
    #   plt.title('Breaking Demand vs. Timestamp')
    #   plt.gca().spines[['top', 'right']].set_visible(False)  # Optional: Removes top and right borders
    #   axes[0,0].set_title("Braking demand vs TimeStamp")

    #   plt.sca(axes[0,1])
    #   sns.boxplot(x=self.bd)
    #   plt.xlabel('All timestamp')
    #   axes[0,1].set_title('Boxplot of Braking Demand')

    #   plt.sca(axes[1,0])
    #   sns.violinplot(x=self.bd)
    #   axes[1,0].set_title('Violin Plot of Braking Demand')

    #   plt.sca(axes[1,1])
    #   plt.plot(self.df['del_s'], self.bd, marker='o', linestyle='-', color='b')
    #   plt.xlabel('del_s (Longitudinal Distance)')
    #   plt.ylabel('Breaking Demand (bd)')
    #   plt.title('Breaking Demand vs. Longitudinal Distance')
    #   plt.grid(True)
    #   plt.gca().spines[['top', 'right']].set_visible(False)
    #   plt.axhline(y=5, color='r', linestyle='--',label='Threshold')
    #   # plt.axvline(x=8, color='green', linestyle='--',label='full breaking')
    #   axes[1,1].set_title("Braking demand vs Longitudinal Distance")
    #   plt.legend()

    #   plt.show()


    def bd_plot(self,save_path="E:\TiHAN\Data_Pipeline\plots"):
      # Ensure bd length matches Timestamp[ms]
      self.brd = (self.df['rel_v'])**2 / (2 * self.df['del_s'])
      self.brd = self.brd[:len(self.df['Timestamp[ms]'])]

      fig, axes = plt.subplots(2, 2, figsize=(10, 10))

      # Plot 1: Braking Demand vs. Timestamp
      axes[0, 0].plot(self.df['Timestamp[ms]'], self.brd)
      axes[0,0].axhline(y=5, color='r', linestyle='--', label='Threshold')
      axes[0, 0].set_xlabel('Timestamp[ms]')
      axes[0, 0].set_ylabel('Braking Demand')
      axes[0, 0].set_title("Braking Demand vs. Timestamp")
      axes[0, 0].spines[['top', 'right']].set_visible(False)

      # Plot 2: Boxplot of Braking Demand
      sns.boxplot(x=self.brd, ax=axes[0, 1])
      axes[0, 1].set_xlabel('Braking Demand')
      axes[0, 1].set_title('Boxplot of Braking Demand')

      # Plot 3: Violin Plot of Braking Demand
      sns.violinplot(x=self.brd, ax=axes[1, 0])
      axes[1, 0].set_title('Violin Plot of Braking Demand')
      axes[1, 0].set_xlabel('Braking Demand')

      # Plot 4: Braking Demand vs. Longitudinal Distance
      axes[1, 1].plot(self.df['del_s'], self.brd, marker='o', linestyle='-', color='b', label='Braking Demand')
      axes[1, 1].axhline(y=5, color='r', linestyle='--', label='Threshold')
      axes[1, 1].set_xlabel('del_s (Longitudinal Distance)')
      axes[1, 1].set_ylabel('Braking Demand')
      axes[1, 1].set_title("Braking Demand vs. Longitudinal Distance")
      axes[1, 1].spines[['top', 'right']].set_visible(False)
      axes[1, 1].grid(True)
      axes[1, 1].legend()

      # Show the plots
      plt.tight_layout()
      plt.savefig(os.path.join(save_path, "breaking_demand_plots.png"))
      print("Breaking demand plot saved")

    #   plt.show()


    def to_csv(self,filename):
      self.df.to_csv(f'{filename}.csv', index=True)
      print(f"DataFrame saved to {filename}")

    def find_peak(self,save_path="E:\TiHAN\Data_Pipeline\plots"):
      from scipy.signal import find_peaks
      x=self.df['rel_v'].values
      peaks, _ = find_peaks(x, height=0,distance=50,prominence=None)
      self.vel_plot()
      plt.scatter(self.df['Timestamp[ms]'].iloc[peaks], x[peaks], color='red')
      plt.savefig(os.path.join(save_path, "peaks_plots.png"))
    #   plt.show()
      print("Peak Index | Peak Value | Peak Timestamp")
      print("-" * 40)
      for idnx ,values , time_stamp in zip(peaks,x[peaks],self.df['Timestamp[ms]'].iloc[peaks]):
        print(f"{idnx} | {values} | {time_stamp}")
      print("All plot saved")





if __name__ == "__main__": 
  data = input("Enter the data location :")
  df1 = vru_test(data) 
