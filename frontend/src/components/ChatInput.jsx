import React, { useRef, useState } from 'react'
import {
  Box,
  Chip,
  IconButton,
  Menu,
  MenuItem,
  TextField,
  Tooltip,
  Typography,
} from '@mui/material'
import AddIcon from '@mui/icons-material/Add'
import ArrowUpwardIcon from '@mui/icons-material/ArrowUpward'
import PhotoCameraIcon from '@mui/icons-material/PhotoCamera'
import StopIcon from '@mui/icons-material/Stop'

/**
 * Chat input bar for sending text, image-only, or multimodal messages.
 */
export default function ChatInput({
  onSend,
  disabled,
  isGenerating,
  onStop,
}) {
  const [text, setText] = useState('')
  const [attachedFile, setAttachedFile] = useState(null)
  const [anchorEl, setAnchorEl] = useState(null)
  const menuOpen = Boolean(anchorEl)
  const fileInputRef = useRef()

  const handleSubmit = (e) => {
    e.preventDefault()
    const trimmed = text.trim()
    if ((!trimmed && !attachedFile) || disabled || isGenerating) return
    onSend(trimmed, attachedFile)
    setText('')
    setAttachedFile(null)
  }

  const openMenu = (e) => setAnchorEl(e.currentTarget)
  const closeMenu = () => setAnchorEl(null)

  return (
    <Box
      component="form"
      onSubmit={handleSubmit}
      sx={{
        width: '100%',
        position: 'relative',
        pb: 1,
      }}
    >
      {attachedFile ? (
        <Box
          sx={{
            mb: 1,
            px: 1,
            display: 'flex',
            alignItems: 'center',
            gap: 1,
            flexWrap: 'wrap',
          }}
        >
          <Chip
            label={attachedFile.name}
            onDelete={() => setAttachedFile(null)}
            size="small"
          />
          <Typography variant="caption" color="text.secondary">
            Add text to combine image + language search, or send image-only.
          </Typography>
        </Box>
      ) : null}

      <TextField
        multiline
        minRows={2}
        variant="outlined"
        fullWidth
        placeholder={attachedFile ? 'Add optional text for this image...' : 'Type your message...'}
        value={text}
        onChange={(e) => setText(e.target.value)}
        disabled={disabled}
        sx={{
          '& .MuiOutlinedInput-root': {
            borderRadius: '16px',
            pb: '3rem',
          },
        }}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            handleSubmit(e)
          }
        }}
      />

      <IconButton
        onClick={openMenu}
        disabled={disabled || isGenerating}
        sx={{
          position: 'absolute',
          bottom: 8,
          left: 8,
        }}
      >
        <AddIcon />
      </IconButton>

      <IconButton
        type={isGenerating ? 'button' : 'submit'}
        onClick={isGenerating ? onStop : undefined}
        disabled={disabled}
        sx={{
          position: 'absolute',
          bottom: 8,
          right: 8,
        }}
      >
        {isGenerating ? <StopIcon /> : <ArrowUpwardIcon />}
      </IconButton>

      <Menu
        anchorEl={anchorEl}
        open={menuOpen}
        onClose={closeMenu}
        anchorOrigin={{ vertical: 'top', horizontal: 'left' }}
        transformOrigin={{ vertical: 'bottom', horizontal: 'left' }}
      >
        <MenuItem
          onClick={() => {
            if (fileInputRef.current) fileInputRef.current.click()
            closeMenu()
          }}
          disableRipple
        >
          <Tooltip title="Attach an image for multimodal or image-only search">
            <IconButton
              size="small"
              sx={{ p: 0.5 }}
              disabled={disabled || isGenerating}
            >
              <PhotoCameraIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </MenuItem>
      </Menu>

      <input
        ref={fileInputRef}
        hidden
        accept="image/*"
        type="file"
        onChange={(e) => {
          if (e.target.files?.[0]) {
            setAttachedFile(e.target.files[0])
            e.target.value = ''
          }
        }}
      />
    </Box>
  )
}
