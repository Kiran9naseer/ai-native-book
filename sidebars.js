// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Module 1: ROS 2 Fundamentals',
      link: {
        type: 'doc',
        id: 'modules/ros2/introduction',
      },
      items: [
        'modules/ros2/architecture',
        'modules/ros2/communication',
        'modules/ros2/urdf-modeling',
        'modules/ros2/ai-integration',
        'modules/ros2/summary',
      ],
    },
    {
      type: 'category',
      label: 'Module 2: Gazebo & Unity',
      link: {
        type: 'doc',
        id: 'modules/gazebo-unity/introduction',
      },
      items: [
        'modules/gazebo-unity/gazebo-physics',
        'modules/gazebo-unity/unity-rendering',
        'modules/gazebo-unity/sensor-integration',
        'modules/gazebo-unity/digital-twin',
      ],
    },
    {
      type: 'category',
      label: 'Module 3: Isaac AI Brain',
      link: {
        type: 'doc',
        id: 'modules/isaac-ai-brain/introduction',
      },
      items: [
        'modules/isaac-ai-brain/isaac-sim',
        'modules/isaac-ai-brain/perception-pipelines',
        'modules/isaac-ai-brain/navigation',
      ],
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action',
      link: {
        type: 'doc',
        id: 'modules/vla/introduction',
      },
      items: [
        'modules/vla/voice-to-action',
        'modules/vla/cognitive-planning',
        'modules/vla/autonomous-humanoid',
      ],
    },
  ],
};

module.exports = sidebars;