// frontend/src/components/Intersection.jsx
import React, { useEffect, useRef } from 'react';
import './Intersection.css';

const Intersection = ({ metrics }) => {
  const canvasRef = useRef(null);
  const vehiclesRef = useRef({});
  const metricsRef = useRef(metrics);

  useEffect(() => {
    metricsRef.current = metrics;
  }, [metrics]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const centerX = width / 2;
    const centerY = height / 2;
    const LANE_WIDTH = 20; // Width of one lane
    const VEHICLE_SIZE = 8;

    // GREEN_PHASE_LANE_MAP defines which lanes have green light for each phase
    const GREEN_PHASE_LANE_MAP = {
      'NS_STRAIGHT_GREEN': [['north', 'straight'], ['south', 'straight'], ['north', 'right'], ['south', 'right']],
      'NS_LEFT_GREEN': [['north', 'left'], ['south', 'left']],
      'EW_STRAIGHT_GREEN': [['east', 'straight'], ['west', 'straight'], ['east', 'right'], ['west', 'right']],
      'EW_LEFT_GREEN': [['east', 'left'], ['west', 'left']],
    };

    const drawIntersection = () => {
      ctx.fillStyle = '#2D3748'; // Dark background
      ctx.fillRect(0, 0, width, height);

      // Draw 6-lane roads (3 lanes each direction)
      ctx.fillStyle = '#4A5568'; // Dark grey for roads
      // Horizontal road (East-West)
      ctx.fillRect(0, centerY - LANE_WIDTH * 3, width, LANE_WIDTH * 6);
      // Vertical road (North-South)
      ctx.fillRect(centerX - LANE_WIDTH * 3, 0, LANE_WIDTH * 6, height);

      // Draw intersection box
      ctx.fillStyle = '#3A475A';
      ctx.fillRect(centerX - LANE_WIDTH * 3, centerY - LANE_WIDTH * 3, LANE_WIDTH * 6, LANE_WIDTH * 6);

      // Draw center yellow dividers (double lines)
      ctx.strokeStyle = '#F1C40F';
      ctx.lineWidth = 2;
      ctx.setLineDash([]);
      
      // Horizontal center divider
      ctx.beginPath();
      ctx.moveTo(0, centerY - 1);
      ctx.lineTo(centerX - LANE_WIDTH * 3, centerY - 1);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(centerX + LANE_WIDTH * 3, centerY - 1);
      ctx.lineTo(width, centerY - 1);
      ctx.stroke();
      
      ctx.beginPath();
      ctx.moveTo(0, centerY + 1);
      ctx.lineTo(centerX - LANE_WIDTH * 3, centerY + 1);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(centerX + LANE_WIDTH * 3, centerY + 1);
      ctx.lineTo(width, centerY + 1);
      ctx.stroke();

      // Vertical center divider
      ctx.beginPath();
      ctx.moveTo(centerX - 1, 0);
      ctx.lineTo(centerX - 1, centerY - LANE_WIDTH * 3);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(centerX - 1, centerY + LANE_WIDTH * 3);
      ctx.lineTo(centerX - 1, height);
      ctx.stroke();
      
      ctx.beginPath();
      ctx.moveTo(centerX + 1, 0);
      ctx.lineTo(centerX + 1, centerY - LANE_WIDTH * 3);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(centerX + 1, centerY + LANE_WIDTH * 3);
      ctx.lineTo(centerX + 1, height);
      ctx.stroke();

      // Draw lane dividers (dashed white lines)
      ctx.strokeStyle = '#FFFFFF';
      ctx.lineWidth = 1;
      ctx.setLineDash([8, 8]);

      // Horizontal lane dividers (2 on each side of center)
      for (let i = 1; i <= 2; i++) {
        const y = centerY - i * LANE_WIDTH;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(centerX - LANE_WIDTH * 3, y);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(centerX + LANE_WIDTH * 3, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }

      for (let i = 1; i <= 2; i++) {
        const y = centerY + i * LANE_WIDTH;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(centerX - LANE_WIDTH * 3, y);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(centerX + LANE_WIDTH * 3, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }

      // Vertical lane dividers (2 on each side of center)
      for (let i = 1; i <= 2; i++) {
        const x = centerX - i * LANE_WIDTH;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, centerY - LANE_WIDTH * 3);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(x, centerY + LANE_WIDTH * 3);
        ctx.lineTo(x, height);
        ctx.stroke();
      }

      for (let i = 1; i <= 2; i++) {
        const x = centerX + i * LANE_WIDTH;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, centerY - LANE_WIDTH * 3);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(x, centerY + LANE_WIDTH * 3);
        ctx.lineTo(x, height);
        ctx.stroke();
      }

      ctx.setLineDash([]);
    };

    const drawTrafficLights = () => {
      if (!metricsRef.current) return;
      
      const greenLanes = GREEN_PHASE_LANE_MAP[metricsRef.current.current_phase_name] || [];
      const isYellow = metricsRef.current.current_phase_name?.includes('YELLOW');
      const redColor = '#E74C3C', yellowColor = '#F1C40F', greenColor = '#2ECC71';

      const signalPositions = {
        north: { left: centerX + LANE_WIDTH * 0.5, straight: centerX + LANE_WIDTH * 1.5, right: centerX + LANE_WIDTH * 2.5, y: centerY - LANE_WIDTH * 3 - 15 },
        south: { left: centerX - LANE_WIDTH * 0.5, straight: centerX - LANE_WIDTH * 1.5, right: centerX - LANE_WIDTH * 2.5, y: centerY + LANE_WIDTH * 3 + 15 },
        east: { left: centerY + LANE_WIDTH * 0.5, straight: centerY + LANE_WIDTH * 1.5, right: centerY + LANE_WIDTH * 2.5, x: centerX + LANE_WIDTH * 3 + 15 },
        west: { left: centerY - LANE_WIDTH * 0.5, straight: centerY - LANE_WIDTH * 1.5, right: centerY - LANE_WIDTH * 2.5, x: centerX - LANE_WIDTH * 3 - 15 },
      };

      Object.keys(signalPositions).forEach(dir => {
        ['left', 'straight', 'right'].forEach(turn => {
          const isMyLaneGreen = greenLanes.some(([d, t]) => d === dir && t === turn);
          let color = redColor;
          if (isMyLaneGreen) {
            color = greenColor;
          } else if (isYellow) {
            const yellowPhase = metricsRef.current.current_phase_name;
            const previousPhase = yellowPhase.replace('YELLOW', 'GREEN');
            const previousGreenLanes = GREEN_PHASE_LANE_MAP[previousPhase] || [];
            if (previousGreenLanes.some(([d, t]) => d === dir && t === turn)) {
              color = yellowColor;
            }
          }
          
          const x = (dir === 'east' || dir === 'west') ? signalPositions[dir].x : signalPositions[dir][turn];
          const y = (dir === 'north' || dir === 'south') ? signalPositions[dir].y : signalPositions[dir][turn];
          
          ctx.fillStyle = '#111';
          ctx.beginPath();
          ctx.arc(x, y, 6, 0, 2 * Math.PI);
          ctx.fill();
          
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(x, y, 4, 0, 2 * Math.PI);
          ctx.fill();
        });
      });
    };

    const drawVehicle = (x, y, direction, vehicleId) => {
      ctx.save();
      ctx.translate(x, y);
      const rotations = { north: 180, south: 0, east: -90, west: 90 };
      ctx.rotate((rotations[direction] || 0) * Math.PI / 180);
      
      const color = `hsl(${(vehicleId * 50) % 360}, 90%, 70%)`;
      ctx.fillStyle = color;
      ctx.fillRect(-VEHICLE_SIZE / 2, -VEHICLE_SIZE, VEHICLE_SIZE, VEHICLE_SIZE * 2);
      
      ctx.fillStyle = 'rgba(0, 0, 0, 0.4)';
      ctx.fillRect(-VEHICLE_SIZE / 2, -VEHICLE_SIZE * 0.5, VEHICLE_SIZE, VEHICLE_SIZE);
      
      ctx.restore();
    };

    const getStartPosition = (direction, turn) => {
      const turnOffsets = { left: 0.5, straight: 1.5, right: 2.5 };
      const offset = turnOffsets[turn] * LANE_WIDTH;

      switch (direction) {
        case 'north': return { x: centerX + offset, y: 0 };
        case 'south': return { x: centerX - offset, y: height };
        case 'east':  return { x: width, y: centerY + offset };
        case 'west':  return { x: 0, y: centerY - offset };
        default: return { x: 0, y: 0 };
      }
    };

    const calculateQueuePosition = (direction, turn, index) => {
      // Stop line is at the edge of the intersection (where vehicles should stop)
      const stopLine = {
        north: (centerY - LANE_WIDTH * 3) / height,
        south: (centerY + LANE_WIDTH * 3) / height,
        east: (centerX + LANE_WIDTH * 3) / width,
        west: (centerX - LANE_WIDTH * 3) / width,
      }[direction];

      const vehicleSpacing = (VEHICLE_SIZE * 2.5) / (['north', 'south'].includes(direction) ? height : width);
      
      // For north and west: vehicles come from low values, queue BEFORE (subtract from) stopLine
      // For south and east: vehicles come from high values, queue AFTER (add to) stopLine
      if (direction === 'north' || direction === 'west') {
        return stopLine - (index * vehicleSpacing);
      } else {
        return stopLine + (index * vehicleSpacing);
      }
    };

    const getPositionOnLane = (laneId, progress, direction) => {
      const startPos = getStartPosition(direction, laneId.split('_')[1]);
      let x = startPos.x;
      let y = startPos.y;

      switch (direction) {
        case 'north':
          y = progress * height;
          break;
        case 'south':
          y = (1 - progress) * height;
          break;
        case 'east':
          x = (1 - progress) * width;
          break;
        case 'west':
          x = progress * width;
          break;
      }
      return { x, y };
    };

    const updateVehicles = () => {
      if (!metricsRef.current || !metricsRef.current.lanes) return;

      const newVehicles = { ...vehiclesRef.current };
      const allLanes = metricsRef.current.lanes;

      // Iterate through directions and turns
      Object.entries(allLanes).forEach(([direction, turns]) => {
        Object.entries(turns).forEach(([turn, laneData]) => {
          if (laneData && laneData.queue) {
            const laneId = `${direction}_${turn}`;
            laneData.queue.forEach((vehicle, index) => {
              const vehicleId = `${laneId}-${vehicle.id}`;
              if (!newVehicles[vehicleId]) {
                newVehicles[vehicleId] = {
                  ...getStartPosition(direction, turn),
                  id: vehicle.id,
                  laneId: laneId,
                  direction: direction,
                  progress: 0,
                  targetProgress: calculateQueuePosition(direction, turn, index),
                };
              } else {
                newVehicles[vehicleId].targetProgress = calculateQueuePosition(direction, turn, index);
              }
            });
          }
        });
      });

      // Remove vehicles that are no longer in any queue
      Object.keys(newVehicles).forEach(vehicleId => {
        const { laneId, id } = newVehicles[vehicleId];
        const [direction, turn] = laneId.split('_');
        const laneData = allLanes[direction]?.[turn];
        if (!laneData || !laneData.queue.some(v => v.id === id)) {
          newVehicles[vehicleId].targetProgress = 1.1;
          setTimeout(() => {
            delete vehiclesRef.current[vehicleId];
          }, 2000);
        }
      });

      vehiclesRef.current = newVehicles;
    };

    const animate = () => {
      const newVehicles = { ...vehiclesRef.current };
      let changed = false;

      Object.keys(newVehicles).forEach(id => {
        const v = newVehicles[id];
        if (Math.abs(v.progress - v.targetProgress) > 0.001) {
          v.progress += (v.targetProgress - v.progress) * 0.1;
          changed = true;
        }
      });

      if (changed) {
        vehiclesRef.current = newVehicles;
      }

      ctx.clearRect(0, 0, width, height);
      drawIntersection();
      drawTrafficLights();

      Object.values(vehiclesRef.current).forEach(v => {
        if (v.progress > 1.1) return;
        const pos = getPositionOnLane(v.laneId, v.progress, v.direction);
        drawVehicle(pos.x, pos.y, v.direction, v.id);
      });

      requestAnimationFrame(animate);
    };

    const intervalId = setInterval(updateVehicles, 500);
    animate();

    return () => {
      clearInterval(intervalId);
    };
  }, []);

  return <canvas ref={canvasRef} width={800} height={800} className="intersection-canvas" />;
};

export default Intersection;
