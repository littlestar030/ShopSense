import React from 'react'
import { Card, CardContent, Typography, Box, Button } from '@mui/material'
import { useTheme, alpha } from '@mui/material/styles'
import ImageSearchIcon from '@mui/icons-material/ImageSearch'
import { motion } from 'framer-motion'

/**
 * Product card with animated entry for displaying recommended products.
 * @param {object} props - Product info fields from backend.
 */
export default function ProductCard({
  id,
  name,
  description,
  price,
  image_path,
  category,
  tags = [],
  use_cases = [],
  features = [],
  onSearchSimilar,
}) {
  const theme = useTheme()
  const backendUrl = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000'
  const titleColor = theme.palette.text.primary
  const descriptionColor = theme.palette.text.secondary
  const mutedColor = theme.palette.text.secondary
  const thumbBg = theme.palette.mode === 'dark' ? '#f5f5f5' : '#fafafa'
  const cardBorder = alpha(theme.palette.divider, 0.9)
  const cardBg = theme.palette.background.paper

  // Build product image URL
  let src
  if (image_path.startsWith('http')) {
    src = image_path
  } else {
    const filename = image_path.startsWith('images/')
      ? image_path.slice('images/'.length)
      : image_path.replace(/^\/+/, '')
    src = `${backendUrl}/images/${filename}`
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 72, scale: 0.9, filter: 'blur(8px)' }}
      animate={{ opacity: 1, y: 0, scale: 1, filter: 'blur(0)' }}
      transition={{ type: 'spring', stiffness: 300, damping: 22, mass: 1, delay: 0.07 }}
      style={{ width: '100%' }}
    >
      <Card
        sx={{
          display: 'flex',
          maxWidth: 720,
          width: '100%',
          m: 1.5,
          boxShadow: theme.palette.mode === 'dark' ? 1 : 2,
          borderRadius: 3,
          alignItems: 'flex-start',
          overflow: 'hidden',
          p: 2.25,
          gap: 2,
          bgcolor: cardBg,
          border: `1px solid ${cardBorder}`,
          '@media (max-width: 720px)': {
            flexDirection: 'column',
            maxWidth: 560,
            p: 1.5,
            gap: 1.5,
          },
        }}
      >
        {/* Product thumbnail */}
        <Box
          sx={{
            width: 132,
            minWidth: 132,
            maxWidth: 132,
            height: 132,
            minHeight: 132,
            maxHeight: 132,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            backgroundColor: thumbBg,
            borderRadius: 2.5,
            overflow: 'hidden',
            flexShrink: 0,
            border: `1px solid ${alpha(theme.palette.divider, 0.8)}`,
            '@media (max-width: 720px)': {
              width: 120,
              minWidth: 120,
              maxWidth: 120,
              height: 120,
              minHeight: 120,
              maxHeight: 120,
            },
          }}
        >
          <img
            src={src}
            alt={name}
            style={{
              width: '100%',
              height: '100%',
              objectFit: 'cover',
              borderRadius: 0,
              display: 'block',
              background: thumbBg,
              imageRendering: 'auto',
            }}
          />
        </Box>

        {/* Product info (right) */}
        <Box sx={{
          display: 'flex',
          flexDirection: 'column',
          flex: 1,
          minHeight: 132,
          justifyContent: 'space-between',
        }}>
          <CardContent sx={{ flex: '1 1 auto', p: 0 }}>
            <Typography
              variant="h6"
              gutterBottom
              sx={{
                fontSize: '1.25rem',
                lineHeight: 1.25,
                fontWeight: 700,
                letterSpacing: '-0.01em',
                color: titleColor,
                mb: 0.6,
              }}
            >
              {name}
            </Typography>
            <Typography
              variant="body2"
              paragraph
              sx={{
                whiteSpace: 'pre-wrap',
                mb: 1.35,
                fontSize: '0.95rem',
                lineHeight: 1.45,
                color: descriptionColor,
              }}
            >
              {description}
            </Typography>
            <Box sx={{ display: 'grid', gap: 0.5 }}>
              <Typography variant="caption" display="block" sx={{ color: mutedColor, fontSize: '0.85rem' }}>
                <Box component="span" sx={{ fontWeight: 700, color: titleColor }}>Category:</Box> {category}
              </Typography>
              <Typography variant="caption" display="block" sx={{ color: mutedColor, fontSize: '0.85rem' }}>
                <Box component="span" sx={{ fontWeight: 700, color: titleColor }}>Use Cases:</Box> {use_cases.slice(0, 3).join(', ')}
              </Typography>
              <Typography variant="caption" display="block" sx={{ color: mutedColor, fontSize: '0.85rem' }}>
                <Box component="span" sx={{ fontWeight: 700, color: titleColor }}>Features:</Box> {features.slice(0, 3).join(', ')}
              </Typography>
              <Typography variant="caption" display="block" sx={{ color: mutedColor, fontSize: '0.85rem' }}>
                <Box component="span" sx={{ fontWeight: 700, color: titleColor }}>Tags:</Box> {tags.slice(0, 5).join(', ')}
              </Typography>
            </Box>
          </CardContent>
          <Box sx={{ pt: 1.25, display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 1, flexWrap: 'wrap' }}>
            <Typography variant="subtitle1" sx={{ fontWeight: 800, color: titleColor }}>
              ${price?.toFixed(2) ?? '--'}
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 1, flexWrap: 'wrap' }}>
              {typeof onSearchSimilar === 'function' ? (
                <Button
                  size="small"
                  variant="outlined"
                  startIcon={<ImageSearchIcon fontSize="small" />}
                  onClick={() => onSearchSimilar({ id, name, image_path })}
                >
                  Search Similar
                </Button>
              ) : null}
            </Box>
          </Box>
        </Box>
      </Card>
    </motion.div>
  )
}
