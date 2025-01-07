import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' #Severe calculation inaccuracies with some GPUs 

import tensorflow as tf #type: ignore
import numpy as np
import acoular as ac
import datetime
import h5py #type: ignore
import csv
from threading import Event, Lock, Thread
from queue import Queue
import time
import queue

from config import ConfigManager

from .SamplesProcessor import LastInOut # TODO


ac.config.global_caching = "none" #type: ignore


class Processor:
    def __init__(self, config, device_index, micgeom_path, results_folder, ckpt_path=None, model_on=False, z=2.0,
                 csm_block_size=256, csm_min_queue_size=256, csm_buffer_size=1000):
        """ Processor for the UMA16 Acoustic Camera
        """
        self.config = config
        
        # Device index for the sound device (UMA16 or other microphone array)
        self.device = device_index
        
        self.save_csv = False
        self.save_h5 = False
        self.log_data = False
        
        # Microphone geometry
        self.mics = ac.MicGeom(from_file=micgeom_path)
        
        # Dimensions of the beamforming grid
        self.x_min, self.x_max = self.config.get('beamforming.xmin'), self.config.get('beamforming.xmax')
        self.y_min, self.y_max = self.config.get('beamforming.ymin'), self.config.get('beamforming.ymax')
        self.z_min, self.z_max = self.config.get('beamforming.zmin'), self.config.get('beamforming.zmax')

        # increment for the grid
        self.increment = self.config.get('beamforming.increment')  
        
        self.beamforming_grid = ac.RectGrid(x_min=self.x_min, x_max=self.x_max, y_min=self.y_min, y_max=self.y_max, z=z, increment=self.increment)
        #self.beamforming_grid_2d = ac.RectGrid(x_min=self.x_min, x_max=self.x_max, y_min=self.y_min, y_max=self.y_max, z_min=self.z_min, z_max=self.z_max, increment=self.increment)
        
        self.grid_dim = (int((self.x_max - self.x_min) / self.increment + 1), int((self.y_max - self.y_min) / self.increment + 1))
        
        # Default target frequency
        self.frequency = self.config.get('beamforming.frequency')
        
        # Locks for adjustable parameters
        self.frequency_lock = Lock()
        self.csm_block_size_lock = Lock()
        self.min_queue_size_lock = Lock()
        self.z = Lock()
        self.result_lock = Lock()
        self.beamforming_result_lock = Lock()
        
        # Path to the model checkpoint
        self.ckpt_path = ckpt_path
        self.model_on = model_on
        
        # Size of the buffer in CSM queue
        self.csm_buffer_size = csm_buffer_size
        
        # Size of one block in CSM
        self.csm_block_size = csm_block_size
        
        # Minimum size of the CSM queue
        self.min_queue_size = csm_min_queue_size
        
        # Number of CSMs to be generated
        self.csm_num = 1
        
        # Shape of the CSM
        self.csm_shape = (int(csm_block_size/2+1), 16, 16)
        
        # Results dictionary for models that will be updated
        self.results = {
            'x': [0],
            'y': [0],
            'z': [0],
            's': [0]
        }
        
        # Results dictionary for beamforming
        base_beamforming_result = np.zeros(self.grid_dim)
        self.beamforming_results = {'results' : base_beamforming_result,
                                    'max_x': [0],
                                    'max_y': [0],
                                    'max_s': [0]}
        
        self.results_folder = results_folder
        self.data_filename, self.results_filename = self._get_result_filenames('model')
        
        self._generators()      
        
    def start_model(self):
        """ Start the model processing
        """
        self._generators()
        print("\nStarting the model.")
        
        # filenames
        self.data_filename, self.results_filename = self._get_result_filenames('model')
        
        # Call functions to setup the model
        self.writeH5.name = f"{self.data_filename}.h5"
        self._setup_model()
        
        if self.log_data:
            self.sample_splitter.register_object(self.fft, self.writeH5)
            
        else:
            self.sample_splitter.register_object(self.fft)
        
        print("Registered objects from sample splitter.")
        
        self._model_threads()
        
        if self.log_data:
            print("Starting Data Saving thread.")
            self.save_time_samples_thread.start()
        
        # Start thread for CSM generation
        print("Starting CSM thread.")
        self.csm_thread.start()
        
        # Start thread for prediction
        print("Starting prediction thread.")
        self.compute_prediction_thread.start()

    def stop_model(self):
        """ Stop the model processing
        """
        print("Stopping model processing.")
        
        # Set Event to stop all inner threads
        self.model_stop_event.set()
        
        # End all threads
        self.csm_thread.join()
        print("CSM thread stopped.")
        
        self.compute_prediction_thread.join() 
        print("Prediction thread stopped.")
        
        if self.log_data:
            self.save_time_samples_thread.join()
            print("Data Saving thread stopped.")

        # Clear the CSM queue
        while not self.csm_queue.empty():
            try:
                self.csm_queue.get_nowait()
            except queue.Empty:
                break
        print("CSM queue cleared.")
        
        if self.log_data:
            self.sample_splitter.remove_object(self.fft, self.writeH5)
            
        else:
            self.sample_splitter.remove_object(self.fft)
            
        print("Removed objects from sample splitter.")
        
    def get_results(self):
        """ Get current results of the model
        """
        # Return a copy of the results safely
        with self.result_lock:
            return self.results.copy()
         
    def _generators(self):
        """ Setup the generators for the process
        """
        print("Setting up generators for the process.")
        
        if self.ckpt_path is None or not self.model_on:
            self.dev = ac.SoundDeviceSamplesGenerator(device=self.device, numchannels=16)
        
        else:
            from .SamplesGenerator import SoundDeviceSamplesGeneratorWithPrecision
            self.dev = SoundDeviceSamplesGeneratorWithPrecision(device=self.device, numchannels=16)
            
        # Turn Volt to Pascal 
        self.source_mixer = ac.SourceMixer(sources=[self.dev],weights=np.array([1/0.0016])) #TODO
        
        # Sample Splitter for parallel processing
        self.sample_splitter = ac.SampleSplitter(source=self.source_mixer, buffer_size=1024) 
        
        # Generator for logging the time data
        self.writeH5 = ac.WriteH5(source=self.sample_splitter, name=f"{self.data_filename}.h5") 
        
        if self.ckpt_path is None or not self.model_on:
            print("No model has been loaded. Model Option will not be availiable.")
        else:
            # Real Fast Fourier Transform
            self.fft = ac.RFFT(source=self.sample_splitter, block_size=256)#self.csm_block_size)
            
            # Cross Power Spectra -> CSM
            self.csm_gen = ac.CrossPowerSpectra(source=self.fft)

            # Index of the target frequency
            self.f_ind = np.searchsorted(self.fft.fftfreq(), self.frequency)

        # Steering Vector
        self.steer = ac.SteeringVector(env=ac.Environment(c=343), grid=self.beamforming_grid, mics=self.mics)
        
        self.lastOut = LastInOut(source=self.sample_splitter) #TODO
        
        self.bf = ac.BeamformerTime(source=self.lastOut, steer=self.steer)
        
        self.filter = ac.FiltOctave(source=self.bf, band=self.frequency, fraction='Third octave')
        
        self.power = ac.TimePower(source=self.filter)
        
        self.bf_out = ac.TimeAverage(source=self.power, naverage=512)
        
    def _save_time_samples(self):
        """ Save the time samples to a H5 file """
        gen = self.writeH5.result(num=self.csm_block_size)
        block_count = 0
        
        while not self.model_stop_event.is_set():
            try:
                next(gen)
                block_count += 1
                
            except StopIteration:
                break
            
        print("Finished saving time samples.")
        print(f"Saved {block_count} blocks.")
        
    def _setup_model(self):
        """ Setup the model for the prediction
        """
        print("Setting up model.")

        self.ref_mic_index = 0
        
        # Load the model
        self.model = tf.keras.models.load_model(self.ckpt_path)
        
    def _model_threads(self):
        """ Threads for the model process
        """
        # Queue for the CSM data
        self.csm_queue = Queue(maxsize=self.csm_buffer_size)
        
        # Event to eventually stop all threads
        self.model_stop_event = Event()

        # Threads
        self.csm_thread = Thread(target=self._csm_generator)
        self.compute_prediction_thread = Thread(target=self._predictor)
        
        if self.log_data:
            self.save_time_samples_thread = Thread(target=self._save_time_samples) 
        
    def _csm_generator(self):
        """ CSM generator thread for the model
        """
        gen = self.csm_gen.result(num=self.csm_num)
        
        while not self.model_stop_event.is_set():
            data = next(gen)

            self.csm_queue.put(data)
            
    def _predictor(self):
        """Prediction thread for the model."""
        while not self.model_stop_event.is_set():
            csm_list = []

            # Warte, bis genügend Daten in der Queue sind
            if self.csm_queue.qsize() < self.min_queue_size:
                time.sleep(0.1)
                continue

            # Hole mehrere Blöcke aus der Queue
            while not self.csm_queue.empty():# and len(csm_list) < 10:  # Batchgröße von 10
                try:
                    data = self.csm_queue.get_nowait()  # Hole Daten ohne zu blockieren
                    csm_list.append(data)
                except queue.Empty:
                    break

            print(f"CSM list size: {len(csm_list)}")
            print("new Result!")

            # Berechne den Durchschnitt der gesammelten Blöcke
            csm_mean = np.mean(csm_list, axis=0)

            # Preprocess the CSM data
            eigmode, csm_norm = self._preprocess_csm(csm_mean)

            # Predict the strength and location
            strength_pred, loc_pred, noise_pred = self.model.predict(eigmode, verbose=0)
            strength_pred = strength_pred.squeeze()

            # Weitere Verarbeitung der Ergebnisse
            strength_pred *= np.real(csm_norm)
            strength_pred = ac.L_p(strength_pred)
            loc_pred = loc_pred.squeeze()

            loc_pred *= np.array([1.0, 1.0, 0.5])[:, np.newaxis]
            loc_pred -= np.array([0.0, 0.0, -1.5])[:, np.newaxis]
            loc_pred[0] = -loc_pred[0]

            # Ergebnisse speichern
            with self.result_lock:
                self.results = {
                    'x': loc_pred[0].tolist(),
                    'y': loc_pred[1].tolist(),
                    'z': loc_pred[2].tolist(),
                    's': strength_pred.tolist()
                }

            self._save_results()


    def _preprocess_csm(self, data):
        """ Preprocess the CSM data
        """
        csm = data.reshape(self.csm_shape)
        csm = csm[self.f_ind].reshape(self.dev.numchannels, self.dev.numchannels)

        # Normalization of the CSM with respect to the reference microphone
        csm_norm = csm[self.ref_mic_index, self.ref_mic_index]
        csm = csm / csm_norm
        csm = csm.reshape(1, 16, 16)

        # Preprocessing
        neig = 8
        evls, evecs = np.linalg.eigh(csm)
        eigmode = evecs[..., -neig:] * evls[:, np.newaxis, -neig:]
        eigmode = np.stack([np.real(eigmode), np.imag(eigmode)], axis=3)
        eigmode = np.transpose(eigmode, [0, 2, 1, 3])
        input_shape = np.shape(eigmode)
        eigmode = np.reshape(eigmode, [-1, input_shape[1], input_shape[2]*input_shape[3]])

        return eigmode, csm_norm
    
    def _get_freq_indicies(self, freqs):
        freq_indices = [np.searchsorted(self.fft.fftfreq(), freq) for freq in freqs]
        return freq_indices

    def start_beamforming(self):
        """ Start the beamforming process
        """
        print("\nStarting beamforming.")
        
        self._generators()
        
        self.data_filename, self.results_filename = self._get_result_filenames('beamforming')
        
        self.writeH5.name = f"{self.data_filename}.h5"
        
        if self.log_data:
            self.sample_splitter.register_object(self.lastOut, self.writeH5)

        else:
            self.sample_splitter.register_object(self.lastOut)
        print(self.beamforming_grid.z)
        
        print("Registered objects from sample splitter.")
        
        self._beamforming_threads()
        
        # Start the beamforming thread
        self.beamforming_thread.start()   
        print("Beamforming thread started.")
        
        # Start the thread for saving time samples
        if self.log_data:
            self.save_time_samples_beamforming_thread.start()
            print("Time data saving thread started.")
    
    def stop_beamforming(self):
        """ Stop the beamforming process
        """
        print("Stopping beamforming.")
        
        # Set the event to stop all threads
        self.beamforming_stop_event.set()

        self.beamforming_thread.join()
        print("Beamforming thread stopped.")
        
        if self.log_data:
            self.save_time_samples_beamforming_thread.join()
            print("Time data saving thread stopped.")
            self.sample_splitter.remove_object(self.lastOut, self.writeH5)
        
        else:
            self.sample_splitter.remove_object(self.lastOut)
            
        print("Removed objects from sample splitter.")
        
    def get_beamforming_results(self):
        """ Get current results of the model
        """
        # Return a copy of the results
        with self.beamforming_result_lock:
            return self.beamforming_results.copy()
        
    def _beamforming_threads(self):
        """ Threads for the beamforming process
        """
        # Event to eventually stop all threads
        self.beamforming_stop_event = Event()
        
        self.beamforming_thread = Thread(target=self._beamforming_generator)
        
        if self.log_data:
            self.save_time_samples_beamforming_thread = Thread(target=self._save_time_samples_beamforming)
        
    def _get_maximum_coordinates(self, data):
        """ Get the maximum coordinates of the beamforming results
        """
        max_val_index = np.argmax(data)
        max_x, max_y = np.unravel_index(max_val_index, data.shape)
        x_coord = self.x_min + max_x * self.increment
        x_coord = -x_coord
        y_coord = self.y_min + max_y * self.increment
        
        return [x_coord], [y_coord]
    
    def _beamforming_generator(self):
        """ Beamforming-Generator """
        gen = self.bf_out.result(num=1)
        count = 0
        
        while not self.beamforming_stop_event.is_set():
            try:
                res = ac.L_p(next(gen))
                res = res.reshape(self.grid_dim)[:,::-1]
                count += 1
                with self.beamforming_result_lock:
                    self.beamforming_results['results'] = res 
                    self.beamforming_results['max_x'], self.beamforming_results['max_y'] = self._get_maximum_coordinates(res)
                    self.beamforming_results['max_s'] = np.max(res)
                
            except StopIteration:
                print("Generator has been stopped.")
                break
            
            except Exception as e:
                print(f"Exception in _beamforming_generator: {e}")
                break
        print(f"Beamforming: Calculated {count} results.")
              
    def _save_time_samples_beamforming(self):
        """ Save the time samples to a H5 file """
        gen = self.writeH5.result(num=512) 
        block_count = 0
        
        while not self.beamforming_stop_event.is_set():
            try:
                next(gen)
                block_count += 1
                
            except StopIteration:
                break
            
        print("Finished saving time samples.")
        print(f"Saved {block_count} blocks.")
              
    def update_z(self, z):
        self.z = z
        self.beamforming_grid = ac.RectGrid(x_min=self.x_min, x_max=self.x_max, y_min=self.y_min, y_max=self.y_max, z=z, increment=self.increment)
        
    def update_frequency(self, frequency):
        """ Update the target frequency for the model
        """
        with self.frequency_lock:
            self.frequency = frequency

        self.f_ind = self.fft.fftfreq() -  self.frequency
        print(f"Frequency updated to {self.frequency} Hz.")
        
    def update_csm_block_size(self, block_size):
        """ Update the block size for the CSM
        """
        with self.csm_block_size_lock:
            self.csm_block_size = int(block_size)
            self.csm_shape = (int(block_size/2+1), 16, 16)
        print(f"CSM block size updated to {self.csm_block_size}.")
        
    def update_min_queue_size(self, min_queue_size):
        """ Update the minimum queue size for the CSM
        """
        with self.min_queue_size_lock:
            self.min_queue_size = min_queue_size
        print(f"Minimum queue size updated to {self.min_queue_size}.")   
        
    def _get_current_timestamp(self):
        """ Get the current timestamp in ISO format
        """
        return datetime.datetime.now().isoformat() 
    
    def _get_result_filenames(self, type):
        """ Get the filenames for the results
        """
        current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
        data_filename = self.results_folder + f'/{current_time}_{type}_time_data'
        result_filename = self.results_folder + f'/{current_time}_{type}_results' 
        
        return data_filename, result_filename
                
    def _save_results(self):
        """ Save the results to a CSV and H5 file
        """
        timestamp = self._get_current_timestamp()
        current_results = self.get_results()
        
        with self.frequency_lock:
            current_frequency = self.frequency
        
        # Save the results to a CSV file
        if self.save_csv:
            csv_filename = self.results_filename + '.csv'
            with open(csv_filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                for x, y, z, s in zip(current_results['x'], current_results['y'], current_results['z'], current_results['s']):
                    writer.writerow([timestamp, current_frequency, x, y, z, s])
        
        # Save the results to a H5 file
        if self.save_h5:
            h5_filename = self.results_filename + '.h5'
            with h5py.File(h5_filename, 'a') as hf:
                if 'x' not in hf:
                    hf.create_dataset('timestamp', data=np.array([timestamp]*len(current_results['x']), dtype='S19'), maxshape=(None,))
                    hf.create_dataset('frequency', data=np.array([current_frequency]*len(current_results['x'])), maxshape=(None,))
                    hf.create_dataset('x', data=np.array(current_results['x']), maxshape=(None,))
                    hf.create_dataset('y', data=np.array(current_results['y']), maxshape=(None,))
                    hf.create_dataset('z', data=np.array(current_results['z']), maxshape=(None,))
                    hf.create_dataset('s', data=np.array(current_results['s']), maxshape=(None,))  
                else:
                    for key in ['x', 'y', 'z', 's']:
                        dataset = hf[key]
                        dataset.resize((dataset.shape[0] + len(current_results[key]),))
                        dataset[-len(current_results[key]):] = current_results[key]
                    
                    timestamp_dataset = hf['timestamp']
                    timestamp_dataset.resize((timestamp_dataset.shape[0] + len(current_results['x']),))
                    timestamp_dataset[-len(current_results['x']):] = [timestamp.encode('utf-8')] * len(current_results['x'])
                    freq_dataset = hf['frequency']
                    freq_dataset.resize((freq_dataset.shape[0] + len(current_results['x']),))
                    freq_dataset[-len(current_results['x']):] = [current_frequency] * len(current_results['x'])
                         
    def _save_beamforming_results(self):
        """ Save the results to a CSV and H5 file
        """
        timestamp = self._get_current_timestamp()
        current_results = self.get_beamforming_results()
        
        with self.frequency_lock:
            current_frequency = self.frequency
        
        # Save the results to a CSV file
        if self.save_csv:
            csv_filename = self.results_filename + '.csv'
            with open(csv_filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                for x, y,  s in zip(current_results['max_x'], current_results['max_y'], current_results['max_z']):
                    writer.writerow([timestamp, current_frequency, x, y,  s])
        
        # Save the results to a H5 file
        if self.save_h5:
            h5_filename = self.results_filename + '.h5'
            with h5py.File(h5_filename, 'a') as hf:
                if 'x' not in hf:
                    hf.create_dataset('timestamp', data=np.array([timestamp]*len(current_results['x']), dtype='S19'), maxshape=(None,))
                    hf.create_dataset('frequency', data=np.array([current_frequency]*len(current_results['x'])), maxshape=(None,))
                    hf.create_dataset('x', data=np.array(current_results['max_x']), maxshape=(None,))
                    hf.create_dataset('y', data=np.array(current_results['max_y']), maxshape=(None,))
                    hf.create_dataset('s', data=np.array(current_results['max_s']), maxshape=(None,))  
                else:
                    for key in ['x', 'y', 's']:
                        dataset = hf[key]
                        dataset.resize((dataset.shape[0] + len(current_results[key]),))
                        dataset[-len(current_results[key]):] = current_results[key]
                    
                    timestamp_dataset = hf['timestamp']
                    timestamp_dataset.resize((timestamp_dataset.shape[0] + len(current_results['x']),))
                    timestamp_dataset[-len(current_results['max_x']):] = [timestamp.encode('utf-8')] * len(current_results['max_x'])
                    freq_dataset = hf['frequency']
                    freq_dataset.resize((freq_dataset.shape[0] + len(current_results['max_x']),))
                    freq_dataset[-len(current_results['max_x']):] = [current_frequency] * len(current_results['max_x'])
    