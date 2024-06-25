from mayavi import mlab
import numpy as np
from scipy import stats
from traits.api import HasTraits, Instance, Button
from traitsui.api import View, Item, Group
from mayavi.core.ui.api import MayaviScene, SceneEditor, MlabSceneModel

class Visualization(HasTraits):
    scene = Instance(MlabSceneModel, ())
    generate_button = Button('Generate New Data')

    def __init__(self):
        HasTraits.__init__(self)
        self.data = np.random.normal(loc=0, scale=1, size=200)
        self.plot_data()

    @mlab.animate(delay=100)
    def plot_data(self):
        mlab.clf()

        # Box plot approximation
        q1, median, q3 = np.percentile(self.data, [25, 50, 75])
        iqr = q3 - q1
        lower_whisker = max(np.min(self.data), q1 - 1.5 * iqr)
        upper_whisker = min(np.max(self.data), q3 + 1.5 * iqr)

        mlab.plot3d([-1, 1], [0, 0], [lower_whisker, lower_whisker], color=(0, 0, 1), tube_radius=0.05)
        mlab.plot3d([-1, 1], [0, 0], [upper_whisker, upper_whisker], color=(0, 0, 1), tube_radius=0.05)
        mlab.plot3d([0, 0], [0, 0], [lower_whisker, upper_whisker], color=(0, 0, 1), tube_radius=0.05)
        
        box = mlab.plot3d([-1, 1, 1, -1, -1], [0, 0, 0, 0, 0], [q1, q1, q3, q3, q1], color=(0, 0, 1), tube_radius=0.05)
        median_line = mlab.plot3d([-1, 1], [0, 0], [median, median], color=(1, 0, 0), tube_radius=0.05)

        # Violin plot approximation
        kde = stats.gaussian_kde(self.data)
        x_range = np.linspace(min(self.data), max(self.data), 100)
        kde_values = kde(x_range)
        max_kde = max(kde_values)

        x = np.array([kde_value / max_kde for kde_value in kde_values])
        y = np.zeros_like(x)
        z = x_range

        mlab.plot3d(x, y, z, color=(0, 1, 0), tube_radius=0.05)
        mlab.plot3d(-x, y, z, color=(0, 1, 0), tube_radius=0.05)

        # Swarm plot approximation
        x = np.random.uniform(-0.5, 0.5, size=len(self.data))
        y = np.random.uniform(-0.5, 0.5, size=len(self.data))
        z = self.data

        pts = mlab.points3d(x, y, z, color=(0.5, 0, 0.5), scale_factor=0.1)

        mlab.text3d(0, 0, np.max(self.data) + 0.5, 'Box, Violin, and Swarm Plot Approximations', scale=0.2)

        yield

    def _generate_button_fired(self):
        self.data = np.random.normal(loc=np.random.uniform(-1, 1), scale=np.random.uniform(0.5, 1.5), size=200)
        self.plot_data()

    view = View(Group(
                    Item('scene', editor=SceneEditor(scene_class=MayaviScene), height=500, width=500, show_label=False),
                    Item('generate_button', show_label=False),
                ),
                resizable=True,
                title='3D Plot Demo'
            )

visualization = Visualization()
visualization.configure_traits()