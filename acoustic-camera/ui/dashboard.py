import threading
import numpy as np
import datetime
from bokeh.layouts import column, layout, row
from bokeh.models import Div, CheckboxGroup, RadioButtonGroup, TextInput, Button, ColumnDataSource #type:ignore
from bokeh.plotting import curdoc, figure
from .plotting import AcousticCameraPlot


class Dashboard:
    def __init__(self, config, processor, model_on=False, alphas=None, video_stream=None, stream_on=False, ):
        
        self.config = config

        # Boolean for stream status
        self.stream_on = stream_on
        
        # Stream Object
        self.video_stream = video_stream
        
        if self.stream_on:
            self.frame_width, self.frame_height = video_stream.frame_width, video_stream.frame_height
            
        else:
            self.frame_width, self.frame_height = self.config.get('layout.video.width'), self.config.get('layout.video.height')

        # Boolean for model status
        self.model_on = model_on
        
        # Start time for the measurement
        self.start_time = 0
        
        # Data Processor object, contains model and beamforming
        self.processor = processor
        
        # Setting model and beamforming threads to None
        self.model_thread = None
        self.beamforming_thread = None
        
        # Method for processing the data, 0 is Deep Learning, 1 is Beamforming
        self.method = 1 # Default is Beamfroming
        
        # Setting up the acoustic camera plot
        self.acoustic_camera_plot = AcousticCameraPlot(
                                        config=self.config,
                                        frame_width=self.frame_width,
                                        frame_height=self.frame_height,
                                        mic_positions=processor.mics.mpos,
                                        alphas=alphas,
                                    )    
        
        self.acoustic_camera_plot.fig.output_backend = "webgl" 

        # Setting up the update intervals
        self.estimation_update_interval = self.config.get('app_settings.estimation_update_interval')
        self.beamforming_update_interval = self.config.get('app_settings.beamforming_update_interval')
        self.camera_update_interval = self.config.get('app_settings.camera_update_interval')
        self.overflow_update_interval = self.config.get('app_settings.stream_update_interval')
        
        # Real coordinates  
        self.real_x = self.config.get('app_default_settings.x')
        self.real_y = self.config.get('app_default_settings.y')
        self.real_z = self.config.get('app_default_settings.z')
        
        if processor.dev is not None:
            # Frequency input field
            self.f_input = TextInput(value=str(self.processor.frequency), title="Frequency (Hz)")
            
            # Real coordinates input fields    
            self.x_input = TextInput(value=str(self.real_x), title="Real X")
            self.y_input = TextInput(value=str(self.real_y), title="Real Y")
            self.z_input = TextInput(value=str(self.real_z), title="Real Z")

            if self.model_on:
                self.csm_block_size_input = TextInput(value=str(self.processor.csm_block_size), title="CSM Block Size")
                self.min_queue_size_input = TextInput(value=str(self.processor.min_queue_size), title="Minimum Queue Size")
                
                # Overflow status text
                self.overflow_status = Div(text="Overflow Status: Unknown", width=300, height=30)
                
                # Start text for the measurement button
                self.cluster_results = RadioButtonGroup(labels=["Show all Results", "Cluster"], active=self.acoustic_camera_plot.cluster) 
            
                # Switching between Deep Learning and Beamforming
                self.method_selector = RadioButtonGroup(labels=["Deep Learning", "Beamforming"], active=self.method)  # 0 is "Deep Learning" as default

            # Coordinates
            self.coordinates_display = Div(text="", width=300, height=100)    
                
            # Plot of the deviation of the estimated position
            self.deviation_cds = ColumnDataSource(data=dict(time=[], x_deviation=[], y_deviation=[], z_deviation=[]))
                
            # Cluster distance input field
            self.cluster_distance_input = TextInput(value=str(self.acoustic_camera_plot.min_cluster_distance), title="Cluster Distance")
            
            # Threshold input field
            self.threshold_input = TextInput(value=str(self.acoustic_camera_plot.threshold), title="Threshold")
                                                        
            # Level display
            self.level_display = Div(text="", width=300, height=100)
        
        else:
            # Text telling there is no Microphone Array connected
            self.no_array_text = Div(text="No Microphone Array connected", width=300, height=100)
            
        # Callbacks
        self.camera_view_callback = None
        self.estimation_callback = None
        self.beamforming_callback = None
        self.overflow_callback = None

        # Setting up the deviation plot
        self._create_deviation_plot()
        
        # Setting up the layout
        self.setup_layout()
        
        # Setting up the callbacks
        self.setup_callbacks()
        
    def update_test_data(self):
            if len(self.x_vals) > 100:  # Begrenzung der Datenpunkte
                self.x_vals = self.x_vals[1:]
                self.y_vals = self.y_vals[1:]
            
            # Neue Daten hinzufügen
            self.x_vals.append(len(self.x_vals))
            self.y_vals.append(np.sin(len(self.x_vals) * 0.1))
            
            self.source.data = {'x': self.x_vals, 'y': self.y_vals}
        
    def get_layout(self):
        """Return the layout of the dashboard
        """
        return self.dashboard_layout

    def setup_layout(self):
        """Setup the layout of the dashboard
        """
        # Styling for the sidebar
        sidebar_style = f"""
            <style>
                #sidebar {{
                    position: fixed;
                    top: {self.config.get('layout.sidebar.y')}px;
                    left: {self.config.get('layout.sidebar.x')}px;
                    height: 100%;
                    width: {self.config.get('layout.sidebar.width')}px;
                    background-color: {self.config.get('ui.sidebar_color')};
                    padding: 0px;
                    box-sizing: border-box;
                    overflow: hidden;
                }}
            </style>
            """
            
        content_layout_style = f"""
            <div style="
                position: fixed; 
                top: {self.config.get('layout.sidebar.y')}px; 
                left: {self.config.get('layout.sidebar.width')}px;  
                width: calc(100% - {self.config.get('layout.sidebar.width')}px); 
                height: 100%; 
                overflow: hidden; 
                box-sizing: border-box;">
            </div>
            """
            
        plot_style = f"""
            <style>
                #plot {{
                    position: fixed;
                    top: {self.config.get('layout.plot.y')}px;
                    left: {self.config.get('layout.plot.x')}px;
                    width: calc(100% - 260px);
                    height: 100%;
                    overflow: hidden;
                    box-sizing: border-box;
                }}
            </style>
            """

        
        # Header "Acoustic Camera"
        header = Div(text=f"<h2 style='color:{self.config.get('ui.font_color')}; font-family:{self.config.get('ui.font')}; '>Acoustic Camera</h2>") #margin-left: 20px
        
        # Checkboxes for origin and mic-geom visibility
        self.checkbox_group = CheckboxGroup(labels=["Show Microphone Geometry", "Show Origin"], active=[0, 1])

        # Measurement button
        # Problem: When stop is pressed to quickly and model has not properly started, error occurs
        self.measurement_button = Button(label="Start")
        
        if self.model_on and self.processor.dev is not None:
            self.sidebar_section = column(
                header,
                self.method_selector,
                self.cluster_results,
                self.cluster_distance_input,
                self.x_input,
                self.y_input,
                self.z_input,
                self.f_input,
                self.csm_block_size_input,
                self.min_queue_size_input,
                self.threshold_input,
                self.checkbox_group,
                self.measurement_button,
                self.overflow_status,
                self.coordinates_display,
                self.level_display
            )
            self.checkbox_group.on_change("active", self.toggle_visibility)
            self.method_selector.on_change('active', self.toggle_method)
            self.cluster_results.on_change('active', self.toggle_cluster)
            
        elif self.processor.dev is not None:
            self.sidebar_section = column(
                header,
                self.x_input,
                self.y_input,
                self.z_input,
                self.f_input,
                self.threshold_input,
                self.checkbox_group,
                self.measurement_button,
                self.coordinates_display,
                self.level_display
            )
            self.checkbox_group.on_change("active", self.toggle_visibility)
            
        else:
            self.sidebar_section = self.no_array_text
       
        # Sidebar Column    
        sidebar = column(
            Div(text=f"{sidebar_style}<div id='sidebar'></div>"),#, width=self.config.get("layout.sidebar.width")),
            self.sidebar_section
        )
        
        # Plot Column
        if self.model_on and self.processor.dev is not None:
            content_layout = column(
                Div(text=f"{content_layout_style}"),
                self.acoustic_camera_plot.fig,
                self.deviation_plot,
            )
        else:
            content_layout = column(
                Div(text=f"{content_layout_style}"),
                self.acoustic_camera_plot.fig,
            )

        # Main dashboard layout
        self.dashboard_layout = layout(
            row(sidebar, content_layout),
            background=self.config.get("ui.background"),
            sizing_mode="scale_height"
        )

    def setup_callbacks(self):
        """Setup the callbacks for the dashboard
        """
        # Callbacks for the real coordinates input fields
        self.x_input.on_change("value", self.update_real_x)
        self.y_input.on_change("value", self.update_real_y)
        self.z_input.on_change("value", self.update_real_z)
        
        # Callbacks for the frequency input field
        self.f_input.on_change("value", self.update_frequency)
        
        if self.model_on:
            # Callbacks for the CSM Block size input field
            self.csm_block_size_input.on_change("value", self.update_csm_block_size)
            
            # Callbacks for the minimum number of CSMs in the buffer
            self.min_queue_size_input.on_change("value", self.update_min_queue_size)
            
            # Callbacks for the min cluster distance input field
            self.cluster_distance_input.on_change("value", self.update_min_cluster_distance)
        
        # Callbacks for the threshold input field
        self.threshold_input.on_change("value", self.update_threshold)
  
        # Callbacks for the measurement button
        self.measurement_button.on_click(self.start_measurement)

        # Start the acoustic camera plot
        self.show_acoustic_camera_plot()

    def toggle_method(self, attr, old, new):
        """Callback for the method selector"""
        # Stop the current method
        self.stop_measurement()
        
        # Remove the periodic callbacks
        if self.estimation_callback is not None:
            curdoc().remove_periodic_callback(self.estimation_callback)
            self.estimation_callback = None
            
        if self.beamforming_callback is not None:
            curdoc().remove_periodic_callback(self.beamforming_callback)
            self.beamforming_callback = None
        
        # Start the new method
        if new == 0:
            print("Wechsel zu Deep Learning")
            self.method = 0
            self.estimation_callback = curdoc().add_periodic_callback(
                self.update_estimations, self.estimation_update_interval)
        
        elif new == 1:
            print("Wechsel zu Beamforming")
            self.method = 1
            self.beamforming_callback = curdoc().add_periodic_callback(
                self.update_beamforming, self.beamforming_update_interval)
            
    def toggle_cluster(self, attr, old, new):
        """Callback for the cluster results selector"""
        self.acoustic_camera_plot.cluster = new
            
    def stop_measurement(self):
        """Stop the current measurement"""
        if self.model_thread is not None:
            self.stop_model()
            self.measurement_button.label = self.config.get("ui.start_text")
        
        if self.beamforming_thread is not None:
            self.stop_beamforming()
            self.measurement_button.label = self.config.get("ui.start_text")
                
    def start_measurement(self):
        """Callback für den Messungs-Button, startet oder stoppt die Messung"""        
        if self.method == 0:
            if self.model_thread is None:
                self.start_model()
                self.start_time = datetime.datetime.now()
                self.measurement_button.label = self.config.get("ui.stop_text")
                self._disable_widgets()
            else:
                self.stop_measurement()
                self.measurement_button.label = self.config.get("ui.start_text")
                self._enable_widgets()
        
        elif self.method == 1:
            if self.beamforming_thread is None:
                self.start_beamforming()
                self.start_time = datetime.datetime.now()
                self.measurement_button.label = self.config.get("ui.stop_text")
                self._disable_widgets()
            else:
                self.stop_measurement()
                self.measurement_button.label = self.config.get("ui.start_text")
                self._enable_widgets()
        
    def update_real_x(self, attr, old, new):
        """Callback for the real x input field
        """
        try:
            x = float(new)
            self.real_x = x
        except ValueError:
            pass
    
    def update_real_y(self, attr, old, new):
        """Callback for the real y input field
        """
        try:
            y = float(new)
            self.real_y = y
        except ValueError:
            pass
        
    def update_real_z(self, attr, old, new):
        """Callback for the real z input field
        """
        try:
            z = float(new)
            self.real_z = z
            self.processor.update_z(z)
        except ValueError:
            pass
        
    def update_frequency(self, attr, old, new):
        """Callback for the frequency input field
        """
        try:
            f = float(new)
            self.processor.update_frequency(f)
        except ValueError:
            pass
        
    def update_csm_block_size(self, attr, old, new):
        """Callback for the csmblocksize input field
        """
        try:
            s = float(new)
            self.processor.update_csm_block_size(s)
        except ValueError:
            pass
        
    def update_min_queue_size(self, attr, old, new):
        """Callback for minqueuesize input field
        """
        try:
            s = float(new)
            self.processor.update_min_queue_size(s)
        except ValueError:
            pass
        
    def update_threshold(self, attr, old, new):
        """Callback for the threshold input field
        """
        try:
            t = float(new)
            self.acoustic_camera_plot.update_threshold(t)
        except ValueError:
            pass
        
    def update_min_cluster_distance(self, attr, old, new):
        """Callback for the min cluster distance input field
        """
        try:
            d = float(new)
            self.acoustic_camera_plot.update_min_cluster_distance(d)
        except ValueError:
            pass
        
    def update_overflow_status(self):
        """Update the overflow status text
        """
        overflow = self.processor.dev.overflow
        status_text = f"Overflow Status: {overflow}"
        self.overflow_status.text = status_text
        
    def _create_deviation_plot(self):
        self.deviation_plot = figure(width=600, height=220, title="Live Deviation Plot")
        self.deviation_plot.line(x='time', y='x_deviation', source=self.deviation_cds, color="blue", legend_label="X Deviation")
        self.deviation_plot.line(x='time', y='y_deviation', source=self.deviation_cds, color="green", legend_label="Y Deviation")
        self.deviation_plot.line(x='time', y='z_deviation', source=self.deviation_cds, color="red", legend_label="Z Deviation")
        self.deviation_plot.background_fill_color = self.config.get("ui.background_color")
        self.deviation_plot.border_fill_color = self.config.get("ui.background_color")
        
    def _get_result_filenames(self):
        """ Get the filenames for the results
        """
        current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        return self.processor.results_folder + f'/{current_time}.png'

    def show_acoustic_camera_plot(self):
        if self.stream_on:
            self.video_stream.start()  
        
        # Deep Learning
        if self.method == 0:
            self.stop_beamforming()
            self.acoustic_camera_plot.beamforming_renderer.visible = False
            self.acoustic_camera_plot.model_renderer.visible = True
        
        # Beamforming
        elif self.method == 1:
            self.stop_model()
            self.acoustic_camera_plot.model_renderer.visible = False
            self.acoustic_camera_plot.beamforming_renderer.visible = True

        if self.stream_on:
            if self.camera_view_callback is None:
                self.camera_view_callback = curdoc().add_periodic_callback(self.update_camera_view, self.camera_update_interval)

    def toggle_mic_visibility(self, visible):
        self.acoustic_camera_plot.toggle_mic_visibility(visible)
            
    def toggle_origin_visibility(self, visible):
        self.acoustic_camera_plot.toggle_origin_visibility(visible)
            
    def toggle_visibility(self, attr, old, new):
        self.toggle_mic_visibility(0 in new)
        self.toggle_origin_visibility(1 in new)
    
    def update_camera_view(self):
        with self.video_stream.read_lock:
            img_data = self.video_stream.img.copy()
        self.acoustic_camera_plot.update_camera_image(img_data)

    def start_model(self):
        if self.model_thread is None:
            self.model_thread = threading.Thread(target=self.processor.start_model, daemon=True)
            self.model_thread.start()
        
        if self.estimation_callback is None:
            self.estimation_callback = curdoc().add_periodic_callback(self.update_estimations, self.estimation_update_interval)
            
        if self.overflow_callback is None:
            self.overflow_callback = curdoc().add_periodic_callback(self.update_overflow_status, self.overflow_update_interval)
    
    def stop_model(self):
        if self.model_thread is not None:
            self.processor.stop_model()
            self.model_thread.join()
            self.model_thread = None
            
        if self.estimation_callback is not None:
            curdoc().remove_periodic_callback(self.estimation_callback)
            self.estimation_callback = None
            
        if self.overflow_callback is not None:
            curdoc().remove_periodic_callback(self.overflow_callback)
            self.overflow_callback = None
            
    def update_estimations(self):
        model_data = self.processor.get_results()
        self.acoustic_camera_plot.update_plot_model(model_data)
        
        x_vals = np.round(model_data['x'], 2)
        y_vals = np.round(model_data['y'], 2)
        z_vals = np.round(model_data['z'], 2)
        
        self.coordinates_display.text = f"X: {x_vals}<br>Y: {y_vals}<br>Z: {z_vals}"
        
        self.level_display.text = f"Level: {model_data['s']}"
        
        x_deviation = np.array(model_data['x']) - self.real_x
        y_deviation = np.array(model_data['y']) - self.real_y
        z_deviation = np.array(model_data['z']) - self.real_z
        
        if self.start_time != 0:
            elapsed_time = (datetime.datetime.now() - self.start_time).total_seconds()
        else:
            elapsed_time = 0 
        
        new_deviation_data = dict(
            time=[elapsed_time],
            x_deviation=[np.mean(x_deviation)],  # Durchschnittliche Abweichung
            y_deviation=[np.mean(y_deviation)],
            z_deviation=[np.mean(z_deviation)]
        )
    
        self.deviation_cds.stream(new_deviation_data, rollover=200)
            
    def start_beamforming(self):
        if self.beamforming_thread is None:
            self.beamforming_thread = threading.Thread(target=self.processor.start_beamforming, daemon=True)
            self.beamforming_thread.start()
        
        if self.beamforming_callback is None:
            self.beamforming_callback = curdoc().add_periodic_callback(self.update_beamforming, self.beamforming_update_interval)
            
    def stop_beamforming(self):
        if self.beamforming_thread is not None:
            self.beamforming_thread.join()
            self.beamforming_thread = None
            self.processor.stop_beamforming()
            
        if self.beamforming_callback is not None:
            curdoc().remove_periodic_callback(self.beamforming_callback)
            self.beamforming_callback = None

    def update_beamforming(self):
        beamforming_data = self.processor.get_beamforming_results()
        self.acoustic_camera_plot.update_plot_beamforming(beamforming_data)
        
        x_val = beamforming_data['max_x'][0]
        y_val = beamforming_data['max_y'][0]
        
        self.coordinates_display.text = f"X: {x_val}<br>Y: {y_val}"
        self.level_display.text = f"Level: {beamforming_data['max_s']}"
        
    def update_beamforming_dot(self):
        beamforming_data = self.processor.get_beamforming_results()
        self.acoustic_camera_plot.update_plot_beamforming_dots(beamforming_data)
        
        x_val = beamforming_data['max_x'][0]
        y_val = beamforming_data['max_y'][0]
        
        self.coordinates_display.text = f"X: {x_val}<br>Y: {y_val}"
        
        self.level_display.text = f"Level: {beamforming_data['max_s']}"
        
        x_deviation = x_val - self.real_x
        y_deviation = y_val - self.real_y
        z_deviation = 0
        
        if self.start_time != 0:
            elapsed_time = (datetime.datetime.now() - self.start_time).total_seconds()
        else:
            elapsed_time = 0 
        
        new_deviation_data = dict(
            time=[elapsed_time],
            x_deviation=[x_deviation], 
            y_deviation=[y_deviation],
            z_deviation=[z_deviation]
        )
    
        self.deviation_cds.stream(new_deviation_data, rollover=200)
        
    def _disable_widgets(self):
        self.f_input.disabled = True
        self.x_input.disabled = True
        self.y_input.disabled = True
        self.z_input.disabled = True
        self.threshold_input.disabled = True
        if self.model_on:
            self.csm_block_size_input.disabled = True
            self.min_queue_size_input.disabled = True
            self.cluster_distance_input.disabled = True
            self.method_selector.disabled = True
            self.cluster_results.disabled = True

    def _enable_widgets(self):
        self.f_input.disabled = False
        self.x_input.disabled = False
        self.y_input.disabled = False
        self.z_input.disabled = False
        self.threshold_input.disabled = False
        if self.model_on:
            self.csm_block_size_input.disabled = False
            self.min_queue_size_input.disabled = False
            self.cluster_distance_input.disabled = False
            self.method_selector.disabled = False
            self.cluster_results.disabled = False

        