// frontend/src/components/Intersection.jsx
import React, { useEffect, useRef } from 'react';
import './Intersection.css';

const Intersection = ({ metrics }) => {
  const canvasRef = useRef(null);
  const vehiclesRef = useRef({});
  const metricsRef = useRef(metrics);
  const animationFrameRef = useRef(null);
  const lastUpdateTimeRef = useRef(Date.now());
  const roadTextureCache = useRef([]);

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
    const LANE_WIDTH = 20;
    const VEHICLE_SIZE = 8;
    const VEHICLE_LENGTH = 16;
    const ANIMATION_SPEED = 0.0035; // 60fps smooth (0.35% per frame ≈ 3s to cross)
    const VEHICLE_SPAWN_OFFSET = 0.12; // Start visible from edge

    // GREEN_PHASE_LANE_MAP defines which lanes have green light for each phase
    const GREEN_PHASE_LANE_MAP = {
      'NS_STRAIGHT_GREEN': [['north', 'straight'], ['south', 'straight'], ['north', 'right'], ['south', 'right']],
      'NS_LEFT_GREEN': [['north', 'left'], ['south', 'left']],
      'EW_STRAIGHT_GREEN': [['east', 'straight'], ['west', 'straight'], ['east', 'right'], ['west', 'right']],
      'EW_LEFT_GREEN': [['east', 'left'], ['west', 'left']],
    };

    const drawIntersection = () => {
      // Subtle animated gradient background
      const time = Date.now() / 10000;
      const bgGradient = ctx.createRadialGradient(
        centerX + Math.sin(time) * 50, 
        centerY + Math.cos(time) * 50, 
        0, centerX, centerY, width
      );
      bgGradient.addColorStop(0, '#2D3748');
      bgGradient.addColorStop(1, '#1A202C');
      ctx.fillStyle = bgGradient;
      ctx.fillRect(0, 0, width, height);

      // Draw roads with subtle asphalt texture
      ctx.fillStyle = '#3A4556';
      ctx.fillRect(0, centerY - LANE_WIDTH * 3, width, LANE_WIDTH * 6);
      ctx.fillRect(centerX - LANE_WIDTH * 3, 0, LANE_WIDTH * 6, height);

      // Road texture (cached for performance)
      for (let i = 0; i < 100; i++) {
        const x = Math.random() * width;
        const y = Math.random() * height;
        const isOnRoad = (y > centerY - LANE_WIDTH * 3 && y < centerY + LANE_WIDTH * 3) || 
                         (x > centerX - LANE_WIDTH * 3 && x < centerX + LANE_WIDTH * 3);
        if (isOnRoad) {
          ctx.fillStyle = `rgba(255, 255, 255, ${Math.random() * 0.015})`;
          ctx.fillRect(x, y, 2, 2);
        }
      }

      // Draw intersection box with subtle gradient
      const gradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, LANE_WIDTH * 4);
      gradient.addColorStop(0, '#424F5F');
      gradient.addColorStop(1, '#3A475A');
      ctx.fillStyle = gradient;
      ctx.fillRect(centerX - LANE_WIDTH * 3, centerY - LANE_WIDTH * 3, LANE_WIDTH * 6, LANE_WIDTH * 6);

      // Center dividers with animated pulsing glow
      const glowIntensity = 5 + Math.sin(Date.now() / 500) * 2;
      ctx.shadowBlur = glowIntensity;
      ctx.shadowColor = '#F1C40F';
      ctx.strokeStyle = '#F1C40F';
      ctx.lineWidth = 2.5;
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

      ctx.shadowBlur = 0;

      // Animated lane dividers (moving dashes)
      const dashOffset = (Date.now() / 40) % 16;
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.6)';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([8, 8]);
      ctx.lineDashOffset = -dashOffset;

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
      ctx.lineDashOffset = 0;
      
      // Draw crosswalk stripes with enhanced visibility and subtle glow
      ctx.fillStyle = 'rgba(255, 255, 255, 0.95)';
      ctx.shadowBlur = 3;
      ctx.shadowColor = 'rgba(255, 255, 255, 0.6)';
      const stripeWidth = 5;
      const stripeGap = 4;
      const crosswalkLength = LANE_WIDTH * 6;
      const crosswalkThickness = 8;
      
      // North crosswalk
      for (let i = 0; i < crosswalkLength; i += stripeWidth + stripeGap) {
        ctx.fillRect(centerX - LANE_WIDTH * 3 + i, centerY - LANE_WIDTH * 3 - crosswalkThickness, stripeWidth, crosswalkThickness);
      }
      // South crosswalk
      for (let i = 0; i < crosswalkLength; i += stripeWidth + stripeGap) {
        ctx.fillRect(centerX - LANE_WIDTH * 3 + i, centerY + LANE_WIDTH * 3, stripeWidth, crosswalkThickness);
      }
      // East crosswalk
      for (let i = 0; i < crosswalkLength; i += stripeWidth + stripeGap) {
        ctx.fillRect(centerX + LANE_WIDTH * 3, centerY - LANE_WIDTH * 3 + i, crosswalkThickness, stripeWidth);
      }
      // West crosswalk
      for (let i = 0; i < crosswalkLength; i += stripeWidth + stripeGap) {
        ctx.fillRect(centerX - LANE_WIDTH * 3 - crosswalkThickness, centerY - LANE_WIDTH * 3 + i, crosswalkThickness, stripeWidth);
      }
      
      ctx.shadowBlur = 0;
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
          
          // Draw signal housing
          ctx.fillStyle = '#1a1a1a';
          ctx.fillRect(x - 8, y - 8, 16, 16);
          
          // Draw signal with glow effect
          ctx.shadowBlur = 8;
          ctx.shadowColor = color;
          ctx.fillStyle = '#111';
          ctx.beginPath();
          ctx.arc(x, y, 6, 0, 2 * Math.PI);
          ctx.fill();
          
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(x, y, 4.5, 0, 2 * Math.PI);
          ctx.fill();
          
          ctx.shadowBlur = 0;
        });
      });
    };

    const drawVehicle = (x, y, direction, vehicleId, turn, progress) => {
      ctx.save();
      ctx.translate(x, y);
      
      // Calculate rotation based on direction and turning progress
      let rotation = { north: 180, south: 0, east: -90, west: 90 }[direction] || 0;
      
      // Get intersection entry point
      const intersectionEntry = {
        north: (centerY - LANE_WIDTH * 3) / height,
        south: (height - (centerY + LANE_WIDTH * 3)) / height,
        east: (width - (centerX + LANE_WIDTH * 3)) / width,
        west: (centerX - LANE_WIDTH * 3) / width
      }[direction];
      
      // Add rotation when turning through intersection with smoothstep
      if (progress > intersectionEntry && progress <= 1.0) {
        const turnProgress = (progress - intersectionEntry) / (1.0 - intersectionEntry);
        const smoothTurn = turnProgress * turnProgress * (3 - 2 * turnProgress); // Smoothstep easing
        
        if (turn === 'left') {
          // Rotate left (counter-clockwise for north/south, adjust for east/west)
          if (direction === 'north') rotation += smoothTurn * 90;
          else if (direction === 'south') rotation -= smoothTurn * 90;
          else if (direction === 'east') rotation += smoothTurn * 90;
          else if (direction === 'west') rotation -= smoothTurn * 90;
        } else if (turn === 'right') {
          // Rotate right (clockwise)
          if (direction === 'north') rotation -= smoothTurn * 90;
          else if (direction === 'south') rotation += smoothTurn * 90;
          else if (direction === 'east') rotation -= smoothTurn * 90;
          else if (direction === 'west') rotation += smoothTurn * 90;
        }
      }
      
      ctx.rotate(rotation * Math.PI / 180);
      
      // Car color based on vehicle ID - more realistic car colors
      const carColors = [
        '#E74C3C', // Red
        '#3498DB', // Blue
        '#2ECC71', // Green
        '#F39C12', // Orange
        '#9B59B6', // Purple
        '#1ABC9C', // Teal
        '#E67E22', // Dark Orange
        '#34495E', // Dark Grey
        '#ECF0F1', // Light Grey
        '#95A5A6', // Silver
      ];
      const color = carColors[vehicleId % carColors.length];
      
      // Car body with depth gradient
      const bodyGradient = ctx.createLinearGradient(-VEHICLE_SIZE/2, 0, VEHICLE_SIZE/2, 0);
      const adjustedColor = adjustBrightness(color, 1.2);
      bodyGradient.addColorStop(0, color);
      bodyGradient.addColorStop(0.5, adjustedColor);
      bodyGradient.addColorStop(1, color);
      ctx.fillStyle = bodyGradient;
      ctx.fillRect(-VEHICLE_SIZE / 2, -VEHICLE_LENGTH / 2, VEHICLE_SIZE, VEHICLE_LENGTH);
      
      // Draw windshield (darker)
      ctx.fillStyle = 'rgba(50, 80, 120, 0.6)';
      ctx.fillRect(-VEHICLE_SIZE / 2, -VEHICLE_LENGTH / 2 + 2, VEHICLE_SIZE, VEHICLE_LENGTH * 0.35);
      
      // Draw headlights with animated brightness
      if (progress > 0.01) {
        const brightness = 0.7 + Math.sin(Date.now() / 150) * 0.15;
        ctx.fillStyle = `rgba(255, 255, 200, ${brightness})`;
        ctx.shadowBlur = 4;
        ctx.shadowColor = 'rgba(255, 255, 200, 0.7)';
        ctx.fillRect(-VEHICLE_SIZE / 3, -VEHICLE_LENGTH / 2 - 1, VEHICLE_SIZE / 6, 2);
        ctx.fillRect(VEHICLE_SIZE / 6, -VEHICLE_LENGTH / 2 - 1, VEHICLE_SIZE / 6, 2);
        ctx.shadowBlur = 0;
      }
      
      // Draw wheels (small black rectangles)
      ctx.fillStyle = '#1a1a1a';
      const wheelOffset = VEHICLE_LENGTH * 0.3;
      ctx.fillRect(-VEHICLE_SIZE / 2 - 1, -wheelOffset, 1.5, 3);
      ctx.fillRect(VEHICLE_SIZE / 2 - 0.5, -wheelOffset, 1.5, 3);
      ctx.fillRect(-VEHICLE_SIZE / 2 - 1, wheelOffset - 3, 1.5, 3);
      ctx.fillRect(VEHICLE_SIZE / 2 - 0.5, wheelOffset - 3, 1.5, 3);
      
      // Add subtle shadow under car
      ctx.fillStyle = 'rgba(0, 0, 0, 0.2)';
      ctx.fillRect(-VEHICLE_SIZE / 2 + 0.5, VEHICLE_LENGTH / 2, VEHICLE_SIZE - 1, 1);
      
      ctx.restore();
    };

    const adjustBrightness = (color, factor) => {
      const hex = color.replace('#', '');
      const r = Math.min(255, Math.floor(parseInt(hex.substr(0, 2), 16) * factor));
      const g = Math.min(255, Math.floor(parseInt(hex.substr(2, 2), 16) * factor));
      const b = Math.min(255, Math.floor(parseInt(hex.substr(4, 2), 16) * factor));
      return `rgb(${r}, ${g}, ${b})`;
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

      const vehicleSpacing = (VEHICLE_LENGTH * 1.8) / (['north', 'south'].includes(direction) ? height : width);
      
      // For north and west: vehicles come from low values, queue BEFORE (subtract from) stopLine
      // For south and east: vehicles come from high values, queue AFTER (add to) stopLine
      if (direction === 'north' || direction === 'west') {
        return stopLine - (index * vehicleSpacing);
      } else {
        return stopLine + (index * vehicleSpacing);
      }
    };

    const getPositionOnLane = (laneId, progress, direction, turn) => {
      const startPos = getStartPosition(direction, turn);
      let x = startPos.x;
      let y = startPos.y;
      
      const intersectionEntry = {
        north: centerY - LANE_WIDTH * 3,
        south: centerY + LANE_WIDTH * 3,
        east: centerX + LANE_WIDTH * 3,
        west: centerX - LANE_WIDTH * 3
      }[direction];
      
      const entryProgress = {
        north: intersectionEntry / height,
        south: (height - intersectionEntry) / height,
        east: (width - intersectionEntry) / width,
        west: intersectionEntry / width
      }[direction];

      // Phase 1: Approach to intersection (0 to entryProgress)
      if (progress <= entryProgress) {
        const phase1Progress = progress / entryProgress;
        switch (direction) {
          case 'north':
            y = phase1Progress * intersectionEntry;
            break;
          case 'south':
            y = height - (phase1Progress * (height - intersectionEntry));
            break;
          case 'east':
            x = width - (phase1Progress * (width - intersectionEntry));
            break;
          case 'west':
            x = phase1Progress * intersectionEntry;
            break;
        }
      } 
      // Phase 2: Through intersection with turning (entryProgress to 1.0)
      else if (progress <= 1.0) {
        const phase2Progress = (progress - entryProgress) / (1.0 - entryProgress);
        const turnOffsets = { left: 0.5, straight: 1.5, right: 2.5 };
        const laneOffset = turnOffsets[turn] * LANE_WIDTH;
        
        if (turn === 'straight') {
          // Straight through
          switch (direction) {
            case 'north':
              x = centerX + laneOffset;
              y = intersectionEntry + (phase2Progress * LANE_WIDTH * 6);
              break;
            case 'south':
              x = centerX - laneOffset;
              y = intersectionEntry - (phase2Progress * LANE_WIDTH * 6);
              break;
            case 'east':
              x = intersectionEntry - (phase2Progress * LANE_WIDTH * 6);
              y = centerY + laneOffset;
              break;
            case 'west':
              x = intersectionEntry + (phase2Progress * LANE_WIDTH * 6);
              y = centerY - laneOffset;
              break;
          }
        } else if (turn === 'left') {
          // Left turn - use arc
          const arcProgress = phase2Progress * Math.PI / 2;
          const radius = LANE_WIDTH * 3 - laneOffset;
          
          switch (direction) {
            case 'north':
              x = centerX + radius * Math.sin(arcProgress);
              y = intersectionEntry + radius * (1 - Math.cos(arcProgress));
              break;
            case 'south':
              x = centerX - radius * Math.sin(arcProgress);
              y = intersectionEntry - radius * (1 - Math.cos(arcProgress));
              break;
            case 'east':
              x = intersectionEntry - radius * (1 - Math.cos(arcProgress));
              y = centerY + radius * Math.sin(arcProgress);
              break;
            case 'west':
              x = intersectionEntry + radius * (1 - Math.cos(arcProgress));
              y = centerY - radius * Math.sin(arcProgress);
              break;
          }
        } else if (turn === 'right') {
          // Right turn - tighter arc
          const arcProgress = phase2Progress * Math.PI / 2;
          const radius = laneOffset;
          
          switch (direction) {
            case 'north':
              x = centerX + laneOffset - radius * (1 - Math.cos(arcProgress));
              y = intersectionEntry + radius * Math.sin(arcProgress);
              break;
            case 'south':
              x = centerX - laneOffset + radius * (1 - Math.cos(arcProgress));
              y = intersectionEntry - radius * Math.sin(arcProgress);
              break;
            case 'east':
              x = intersectionEntry - radius * Math.sin(arcProgress);
              y = centerY + laneOffset - radius * (1 - Math.cos(arcProgress));
              break;
            case 'west':
              x = intersectionEntry + radius * Math.sin(arcProgress);
              y = centerY - laneOffset + radius * (1 - Math.cos(arcProgress));
              break;
          }
        }
      }
      // Phase 3: Exit (progress > 1.0)
      else {
        const exitProgress = (progress - 1.0) / 0.1;
        switch (direction) {
          case 'north':
            y = centerY + LANE_WIDTH * 3 + exitProgress * 100;
            break;
          case 'south':
            y = centerY - LANE_WIDTH * 3 - exitProgress * 100;
            break;
          case 'east':
            x = centerX - LANE_WIDTH * 3 - exitProgress * 100;
            break;
          case 'west':
            x = centerX + LANE_WIDTH * 3 + exitProgress * 100;
            break;
        }
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
                  turn: turn,
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

      // Remove vehicles that are no longer in any queue (cleared by green light)
      Object.keys(newVehicles).forEach(vehicleId => {
        const { laneId, id } = newVehicles[vehicleId];
        const [direction, turn] = laneId.split('_');
        const laneData = allLanes[direction]?.[turn];
        if (!laneData || !laneData.queue.some(v => v.id === id)) {
          // Immediately remove vehicle once it's been cleared
          delete newVehicles[vehicleId];
        }
      });

      vehiclesRef.current = newVehicles;
    };

    const animate = () => {
      const now = Date.now();
      const deltaTime = Math.min((now - lastUpdateTimeRef.current) / 16.67, 3); // Cap at 3x for lag spikes
      lastUpdateTimeRef.current = now;

      const newVehicles = { ...vehiclesRef.current };
      let changed = false;

      Object.keys(newVehicles).forEach(id => {
        const v = newVehicles[id];
        if (Math.abs(v.progress - v.targetProgress) > 0.001) {
          // Smooth interpolation with deltaTime compensation
          const smoothSpeed = 0.03 * deltaTime;
          v.progress += (v.targetProgress - v.progress) * Math.min(smoothSpeed, 0.15);
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
        const pos = getPositionOnLane(v.laneId, v.progress, v.direction, v.turn);
        drawVehicle(pos.x, pos.y, v.direction, v.id, v.turn, v.progress);
      });

      animationFrameRef.current = requestAnimationFrame(animate);
    };

    const intervalId = setInterval(updateVehicles, 1000);
    animate();

    return () => {
      clearInterval(intervalId);
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, []);

  return <canvas ref={canvasRef} width={800} height={800} className="intersection-canvas" />;
};

export default Intersection;
