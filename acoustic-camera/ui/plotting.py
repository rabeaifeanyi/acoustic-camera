from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Arrow, VeeHead, ColorBar, LinearColorMapper #type:ignore
from bokeh.palettes import Turbo256
from bokeh.transform import linear_cmap
import numpy as np
from scipy.spatial.distance import pdist, squareform #type:ignore


class AcousticCameraPlot:
    def __init__(self, config, frame_width, frame_height, mic_positions, alphas, max_level=90):
        
        self.config = config
        
        self.threshold = self.config.get('app_default_settings.threshold')
        self.Z = self.config.get('app_default_settings.z')
        self.min_distance = self.config.get('app_default_settings.min_distance')
        
        # Set the frame width and height
        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)
        
        self.cluster_distance = self.config.get('app_default_settings.cluster_distance')
        self.cluster = 1
   
        self.max_level = max_level
        
        # Array with microphone positions
        self.mic_positions = mic_positions
        
        # Data source for the camera image
        self.camera_cds = ColumnDataSource({'image_data': []})

        # Data source for the microphone positions
        self.mic_cds = ColumnDataSource(data=dict(x=[], y=[]))
        
        # Arrow for the origin
        self.arrow_x = None
        self.arrow_y = None
        
        # Camera view angles
        self.alpha_x, self.alpha_y = alphas
        
        # Calculate the view range
        self.xmin, self.xmax, self.ymin, self.ymax = self.calculate_view_range(self.Z)
        
        # Point sizes for the model data
        self.min_point_size, self.max_point_size = 5,20
        
        # Data sources for the model data
        self.model_cds = ColumnDataSource(data=dict(x=[], y=[], z=[], s=[], sizes=[]))
        
        # Data source for the beamforming data
        self.beamforming_cds = ColumnDataSource({'beamformer_data': []})    
        self.beamforming_dot_cds = ColumnDataSource(data=dict(x=[], y=[]))
        
        self.x_min, self.y_min = self.config.get("beamforming.xmin"), self.config.get("beamforming.ymin")
        self.x_max, self.y_max = self.config.get("beamforming.xmax"), self.config.get("beamforming.ymax")
        self.dx = self.x_max - self.x_min
        self.dy = self.y_max - self.y_min
        
        self.bar_low, self.bar_high = self.threshold, self.max_level
        
        self.fig, self.second_view = self._create_plot()

    def update_view_range(self, Z):
        self.xmin, self.xmax, self.ymin, self.ymax = self.calculate_view_range(Z)
        self.fig.x_range.start = self.xmin
        self.fig.x_range.end = self.xmax
        self.fig.y_range.start = self.ymin
        self.fig.y_range.end = self.ymax
        
    def update_threshold(self, threshold):
        self.threshold = threshold
        
    def update_min_cluster_distance(self, distance):
        self.cluster_distance = distance

    def calculate_view_range(self, Z):
        xmax = Z * np.tan(self.alpha_x / 2)
        xmin = -xmax
        ymax = Z * np.tan(self.alpha_y / 2)
        ymin = -ymax
        return xmin, xmax, ymin, ymax

    def update_plot_model(self, model_data):
        self.model_renderer.visible = True
        self.model_shadow_renderer.visible = True
        self.beamforming_renderer.visible = False
        
        x = np.array(model_data['x'])
        y = np.array(model_data['y'])
        z = np.array(model_data['z'])
        s = np.array(model_data['s'])
        
        mask = s >= self.threshold
        
        x, y, z, s = x[mask], y[mask], z[mask], s[mask]
        
        if self.cluster and len(x) > 0:
            x, y, z, s = self._cluster_points(x, y, z, s)

        self.model_cds.data = dict(x=x, y=y, z=z, s=s)
        
    def _cluster_points(self, x_list, y_list, z_list, s_list):
        points = np.array(list(zip(x_list, y_list, z_list)))
        dist_matrix = squareform(pdist(points))
        close_points = dist_matrix < self.cluster_distance
        
        groups = []
        new_strengths = []
        visited = np.zeros(len(points), dtype=bool)

        for i in range(len(points)):
            if not visited[i]:
                group_indices = np.where(close_points[i])[0]
                group_points = points[group_indices]
                group_mean = np.mean(group_points, axis=0)
                groups.append(group_mean)
                group_strength = np.mean(np.array(s_list)[group_indices])
                new_strengths.append(group_strength)
                visited[group_indices] = True

        groups = np.array(groups)
        x_result = groups[:, 0]
        y_result = groups[:, 1]
        z_result = groups[:, 2]
        
        return x_result, y_result, z_result , new_strengths
    
    def update_plot_beamforming(self, results):
        self.model_renderer.visible = False
        self.model_shadow_renderer.visible = False
        self.beamforming_renderer.visible = True

        beamforming_map = results['results']
        beamforming_map_filt = np.where(
            beamforming_map >= self.threshold,
            beamforming_map,                    
            np.nan       
        )
        
        self.beamforming_cds.data = {'beamformer_data': [beamforming_map_filt]}

    def update_plot_beamforming_dots(self, results):
        self.model_renderer.visible = False
        self.model_shadow_renderer.visible = False
        max_x, max_y = results['max_x'], results['max_y']
        self.beamforming_dot_cds.data = dict(x=max_x, y=max_y)

    def update_camera_image(self, img):
        self.camera_cds.data['image_data'] = [img]

    def toggle_mic_visibility(self, visible):
        if visible:
            self.mic_cds.data = dict(x=self.mic_positions[0], y=self.mic_positions[1])
        else:
            self.mic_cds.data = dict(x=[], y=[])

    def toggle_origin_visibility(self, visible):
        if self.arrow_x and self.arrow_y:
            self.arrow_x.visible = visible
            self.arrow_y.visible = visible
            
    def _create_base_fig(self):
        fig = figure(
            tools="",
            width=900, # Attention! Param frame_width causes problems when embedded with flask
            height=600,
            x_range=(self.xmin, self.xmax), 
            y_range=(self.ymin, self.ymax),
            output_backend='webgl'
        )

        fig.image_rgba(
            image='image_data', 
            x=self.xmin, 
            y=self.ymin, 
            dw=(self.xmax-self.xmin), 
            dh=(self.ymax-self.ymin), 
            source=self.camera_cds, 
            alpha=self.config.get('ui.video_alpha')
        )
        
        self.mic_cds.data = dict(x=self.mic_positions[0], y=self.mic_positions[1])
        
        fig.scatter(
            x='x', 
            y='y',
            marker='circle', 
            size=self.config.get('ui.mic_size'), 
            color=self.config.get('ui.mic_color'), 
            line_color=self.config.get('ui.mic_line_color'),
            alpha=self.config.get('ui.mic_alpha'), 
            source=self.mic_cds
        )
        
        self.arrow_x = Arrow(
            end=VeeHead(size=self.config.get('ui.origin_head_size'),fill_color=self.config.get('ui.origin_color'), line_color=self.config.get('ui.origin_color')), 
            x_start=0, 
            y_start=0, 
            x_end=self.config.get('ui.origin_length'), 
            y_end=0, 
            line_width=self.config.get('ui.origin_line_width'),
            line_color=self.config.get('ui.origin_color')
        )
        
        fig.add_layout(self.arrow_x)
        
        self.arrow_y = Arrow(
            end=VeeHead(size=self.config.get('ui.origin_head_size'), fill_color=self.config.get('ui.origin_color'), line_color=self.config.get('ui.origin_color')), 
            x_start=0, 
            y_start=0, 
            x_end=0, 
            y_end=self.config.get('ui.origin_length'), 
            line_width=self.config.get('ui.origin_line_width'),
            line_color=self.config.get('ui.origin_color')
        )
        
        fig.add_layout(self.arrow_y)
        
        fig.xaxis.visible = self.config.get('layout.plot.axis') 
        fig.yaxis.visible = self.config.get('layout.plot.axis') 

        if not self.config.get('layout.plot.grid'):
            fig.xgrid.grid_line_color = None
            fig.ygrid.grid_line_color = None
        
        fig.background_fill_alpha = 0
        fig.border_fill_alpha = 0
        fig.outline_line_alpha = 0
    
        return fig
    
    def _create_base_second_view(self):
        second_view = figure(
            title="View from top",
            tools="",
            width=self.config.get("layout.second_plot.width"),
            height=self.config.get("layout.second_plot.height"),
            x_range=(self.xmin, self.xmax), 
            y_range=(0.0, 2.5),
            output_backend='webgl',
            x_axis_label='x distance [m]',
            y_axis_label='z distance [m]'
        )
        
        second_view.xaxis.visible = self.config.get('layout.second_plot.axis') 
        second_view.yaxis.visible = self.config.get('layout.second_plot.axis') 

        if not self.config.get('layout.second_plot.grid'):
            second_view.xgrid.grid_line_color = None
            second_view.ygrid.grid_line_color = None
        
        return second_view
        
    def _create_plot(self):
        fig = self._create_base_fig()
        
        second_view = self._create_base_second_view()
        
        self.color_mapper = linear_cmap(
            's', 
            Turbo256, 
            self.bar_low, 
            self.bar_high, 
            nan_color="white"
        )
        
        self.model_shadow_renderer = fig.scatter(
            x='x', 
            y='y',
            marker='circle', 
            size=self.config.get('ui.shadow_size'), 
            color=self.config.get('ui.shadow_color'),
            alpha=self.config.get('ui.shadow_alpha'), 
            source=self.model_cds
        )
        
        self.model_renderer = fig.scatter(
            x='x', 
            y='y',
            marker='circle', 
            size=self.config.get('ui.dot_size'), 
            color=self.color_mapper,
            alpha=self.config.get('ui.dot_alpha'), 
            source=self.model_cds
        )
        
        self.second_model_renderer = second_view.scatter(
            x='x', 
            y='z',
            marker='circle', 
            size=self.config.get('ui.dot_size'), 
            color=self.color_mapper,
            alpha=self.config.get('ui.dot_alpha'), 
            source=self.model_cds
        )
        
        self.b_color_mapper = LinearColorMapper(
            palette=Turbo256, 
            low=self.bar_low, 
            high=self.bar_high, 
            nan_color="white"
        )
       
        self.beamforming_renderer = fig.image(
            image='beamformer_data',
            x=self.x_min,
            y=self.y_min,
            dw=self.dx,
            dh=self.dy,
            source=self.beamforming_cds,
            color_mapper=self.b_color_mapper,
            alpha=0.5
        )
        
        self.beamforming_plot = fig.scatter(
            x = 'x',
            y = 'y',
            marker='circle',
            source=self.beamforming_dot_cds,
        )
        
        color_bar = ColorBar(
            color_mapper=self.color_mapper['transform'], 
            label_standoff=12, 
            width=8, 
            location=(0, 0),
            background_fill_color=self.config.get('ui.background_color'),
        )

        fig.add_layout(color_bar, 'right')  
        
        return fig, second_view
    