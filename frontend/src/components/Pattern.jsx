// File: src/components/Pattern.jsx

import React from 'react'
import styled from 'styled-components'

const StyledWrapper = styled.div`
  .pattern-container {
    width: 100%;
    min-height: 100vh;
    position: relative;
    overflow: hidden;
    background: #0b2c56; /* Fallback color */
  }

  .pattern-container::before {
    content: "";
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, #1c3d5a 10%, transparent 20%),
      radial-gradient(circle, transparent 10%, #1c3d5a 20%);
    background-size: 30px 30px;
    animation: moveBackground 10s linear infinite;
  }

  @keyframes moveBackground {
    0% {
      transform: translate(0, 0);
    }
    100% {
      transform: translate(20%, 20%);
    }
  }
`

export default function Pattern({ children }) {
  return (
    <StyledWrapper>
      <div className="pattern-container">
        {children}
      </div>
    </StyledWrapper>
  )
}
