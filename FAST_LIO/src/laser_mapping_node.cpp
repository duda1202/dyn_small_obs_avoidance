/* 
© 2022 Robotics 88
Author: Erin Linebarger <erin@robotics88.com> 
*/

#include "laser_mapping.h"

int main(int argc, char** argv)
{
  ros::init(argc, argv, "laser_mapping");
  if (ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME,
                                     ros::console::levels::Debug)) {
    ros::console::notifyLoggerLevelsChanged();
  }

  ros::NodeHandle node;
  laserMapping::LaserMapping laserMapping(node);
  ros::spin();

  return 0;
}